



import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

file = '../input/kyphosis.csv'

df = pd.read_csv(file)
df.head()
df.info()
sns.pairplot(df,hue = 'Kyphosis',size = 3,markers=["o", "D"])
from sklearn.model_selection import train_test_split
X = df.drop(['Kyphosis'],axis = 1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))

print('\n')

print(classification_report(y_test,rfc_pred))