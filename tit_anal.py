# Titanic Survival Prediction Project
# Created by: Vaishnavi Molabanti

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------

titanic_data = pd.read_csv("titanic.csv")

print("First 5 rows of dataset:")
print(titanic_data.head())

# -----------------------------------------
# STEP 2: Basic Information
# -----------------------------------------

print("\nDataset Information:")
print(titanic_data.info())

print("\nMissing Values:")
print(titanic_data.isnull().sum())

# -----------------------------------------
# STEP 3: Data Cleaning
# -----------------------------------------

# Fill missing age values with average age
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

# Fill missing embarked values
titanic_data['Embarked'] = titanic_data['Embarked'].fillna('S')

# Remove Cabin column because many values are missing
titanic_data = titanic_data.drop(columns=['Cabin'])

# Convert Gender into numbers
# male = 0, female = 1
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked values into numbers
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nCleaned Dataset:")
print(titanic_data.head())

# -----------------------------------------
# STEP 4: Data Visualization
# -----------------------------------------

# Survival count graph
survival_count = titanic_data['Survived'].value_counts()

plt.figure(figsize=(6,4))
survival_count.plot(kind='bar')

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.show()

# Gender survival graph
gender_survival = titanic_data.groupby('Sex')['Survived'].mean()

plt.figure(figsize=(6,4))
gender_survival.plot(kind='bar')

plt.title("Average Survival Based on Gender")
plt.xlabel("Gender (0 = Male, 1 = Female)")
plt.ylabel("Average Survival Rate")

plt.show()

# -----------------------------------------
# STEP 5: Selecting Features
# -----------------------------------------

selected_features = titanic_data[
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
]

target = titanic_data['Survived']

# -----------------------------------------
# STEP 6: Split Dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    selected_features,
    target,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------
# STEP 7: Train Model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------
# STEP 8: Predictions
# -----------------------------------------

predictions = model.predict(X_test)

# -----------------------------------------
# STEP 9: Accuracy
# -----------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# -----------------------------------------
# STEP 10: Save Cleaned Dataset
# -----------------------------------------

titanic_data.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")