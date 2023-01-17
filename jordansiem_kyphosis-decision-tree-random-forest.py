import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv("../input/kyposis/kyphosis.csv")
df.head()
#Child data - Age is in months, Number is number of vertbrae affected, start is where in spine started
sns.pairplot(df,hue='Kyphosis')
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis=1)
y= df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
#Fitting to model
#Kyphosis present?
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#Mislabeling 6
#RAndom forest try....
from sklearn.ensemble import RandomForestClassifier

#Ensemble of decision trees
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
#Random forest could have done better with a larger dataset. Data is already pretty absent.
#Random forest is first quick choice and see what is actually possible with the data.
from sklearn.ensemble import RandomForestClassifier
#Ensemble of decision trees
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))