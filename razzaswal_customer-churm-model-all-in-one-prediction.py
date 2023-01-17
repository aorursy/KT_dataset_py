import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
data = pd.read_csv('../input/customer-churn-dataset/Churn_Modelling.csv')
data.head()
data.shape
data.columns
data.isnull().sum()
numerical_columns = data.select_dtypes(['int64', 'float64']).columns
x = data[numerical_columns]
y = data['Exited']
from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=20)
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(),KNeighborsClassifier()]
for model in models:
    new_model = model.fit(x, y)
    predictions = new_model.predict(x_test)
    score = accuracy_score(predictions, y_test)
    print("Predictions is ->", score)
for model in models:
    new_model = model.fit(x, y)
    predictions = new_model.predict(x_test)
    matrix = confusion_matrix(predictions, y_test)
    print("Predictions is ->", matrix)
data.columns
sns.set()
sns.countplot('Exited', hue='Gender', data=data)
sns.countplot('Exited', hue='IsActiveMember', data=data)
sns.countplot('Tenure', data=data)
sns.countplot('IsActiveMember', hue='Exited', data=data)
