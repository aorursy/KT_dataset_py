# Disabling warnings

import warnings

warnings.simplefilter("ignore")



# Import main libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
import os

print(os.listdir('../input'))
df = pd.read_csv('../input/weatherAUS.csv', index_col='Date', parse_dates=True)
# Show first 5 rows of our data

df.head()
df.index
# Show last 5 rows of our data

df.tail()
# Show size of our data

df.shape
# check for any NAN Data

df.isnull().sum()
df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM'],axis=1)
# Handling missing data

df.dropna(axis=0, subset=['Rainfall'], inplace=True)

#df.fillna({'Evaporation':-99999, 'Sunshine':-99999, 'Cloud9am':-99999, 'Cloud3pm':-99999}, inplace=True)

df.fillna(method='ffill', inplace=True)
# Handling Yes or No data

df.RainTomorrow.replace({'No':0, 'Yes':1}, inplace=True)

df.RainToday.replace({'No':0, 'Yes':1}, inplace=True)
df.info()
objects_cols = df.select_dtypes("object")

objects_cols;
objects_cols = df.select_dtypes("object")





from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for col in objects_cols.columns:

    df[col] = le.fit_transform(df[col])
df.head()
from sklearn.neighbors import KNeighborsClassifier # K Neighbors Classifier Algo.

from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier Algo.

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # To get models info.

from sklearn.model_selection import train_test_split # To split data
X = df.drop(['RainTomorrow'], 1).values # Set our features

y = df['RainTomorrow'].values # Set labels
# Splitting  up data, seting 80% for train and 20% for test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier().fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Accuracy :',acc)

print('=' * 60, '\n')

print(classification_report(y_test, y_pred, target_names=['It Will Not Rain', 'It Will Rain'])) 
model = LogisticRegression().fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Accuracy :',acc)

print('=' * 60, '\n')

print(classification_report(y_test, y_pred, target_names=['It Will Not Rain', 'It Will Rain'])) 
model = RandomForestClassifier().fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Accuracy :',acc)

print('=' * 60, '\n')

print(classification_report(y_test, y_pred, target_names=['It Will Not Rain', 'It Will Rain'])) 
# Confusion matrix

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True)

plt.title('A confusion matrix showing the frequency of misclassifications by our classifier')

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()
from xgboost import XGBClassifier



model = XGBClassifier(learning_rate=0.05)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

acc