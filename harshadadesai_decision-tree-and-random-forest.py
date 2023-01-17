import pandas as pd

import numpy as np 

import matplotlib as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("../input/kyphosis-dataset/kyphosis.csv")
df.head()
df.describe()

df.info()
sns.pairplot(df, hue='Kyphosis', palette='Set1')

from sklearn.model_selection import train_test_split

X=df.drop("Kyphosis",axis=1)

y=df["Kyphosis"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred=dt.predict(X_test)

pred
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pred_rf=rf.predict(X_test)
pred_rf
print(confusion_matrix(y_test,pred_rf))

print(classification_report(y_test,pred_rf))