import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

%matplotlib inline
sns.set_style('whitegrid')
data = pd.read_csv('../input/diabetesdataset/diabetes.csv')
data.head()
data.info()
sns.pairplot(data)
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.distplot(data['BMI'])
sns.distplot(data['SkinThickness'])
sns.countplot(data['Outcome'])
sns.heatmap(data.corr(), cmap='Blues')
X = data.drop('Outcome', axis = 1)

y= data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
lm = LogisticRegression()
lm.fit(X_train, y_train)
pred = lm.predict(X_test)
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data.drop('Outcome',axis=1))
scaled_features = scaler.transform(data.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])

df_feat.head()
df_feat.shape
S_X_train, S_X_test, S_y_train, S_y_test = train_test_split(scaled_features,data['Outcome'],

                                                    test_size=0.30)
slm = LogisticRegression()
slm.fit(S_X_train, S_y_train)
predict = slm.predict(S_X_test)
confusion_matrix(S_y_test, predict)
print(classification_report(S_y_test, predict))