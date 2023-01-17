# Importing all necessary libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.info()
df.describe()
df.shape
df.dtypes
df.isna().sum()
df['Attrition'].value_counts()
import seaborn as sns

sns.countplot(df['Attrition'])
plt.subplots(figsize=(12,4))

sns.countplot(x='Age', hue='Attrition', data=df, palette = 'colorblind')
df['StandardHours'].unique()
df = df.drop("Over18", axis=1)

df = df.drop("EmployeeNumber", axis=1)

df = df.drop("StandardHours", axis=1)

df = df.drop("EmployeeCount", axis=1)
df.corr()
plt.figure(figsize=(14,14))

sns.heatmap(df.corr(), annot=True, fmt='.0%' )
from sklearn.preprocessing import LabelEncoder



for column in df.columns:

    if df[column].dtype == np.number:

        continue

    df[column] = LabelEncoder().fit_transform(df[column])
df['Age_Years'] = df['Age'] 
df = df.drop('Age', axis=1)
df.head()
X = df.iloc[:, 1:df.shape[1]].values

y = df.iloc[:, 0].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

forest.fit(X_train, y_train)
forest.score(X_train, y_train)
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, forest.predict(X_test))



TN = cm[0][0]

TP = cm[1][1]

FN = cm[1][0]

FP = cm[0][1]



print(cm)

print('Model Testing Accuracy = {}'.format((TP + TN / (TP + TN + FN + FP))))