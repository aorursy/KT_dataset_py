
import numpy as np 
import pandas as pd 


import os


df=pd.read_csv('../input/train.csv')
df=pd.get_dummies(data=df,columns=['Sex','Embarked'])
df=df.drop(['Name'],axis=1)
df=df.drop(['Cabin'],axis=1)
df=df.drop(['Ticket'],axis=1)
df=df.drop(['PassengerId'],axis=1)

df=df.fillna(df.median())
Y=df['Survived']
X=df.drop(['Survived'],axis=1)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X,Y)
df=pd.read_csv('../input/train.csv')
df=pd.get_dummies(data=df,columns=['Sex','Embarked'])
df=df.drop(['Name'],axis=1)
df=df.drop(['Cabin'],axis=1)
df=df.drop(['Ticket'],axis=1)
idd=df['PassengerId']
df=df.drop(['PassengerId'],axis=1)
#df.fillna(df.median())
df=df.fillna(df.median())
Y=df['Survived']
X=df.drop(['Survived'],axis=1)
y_pred=clf.predict(X)
y_pred_df=pd.DataFrame(y_pred)
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y, y_pred)
print(acc)
