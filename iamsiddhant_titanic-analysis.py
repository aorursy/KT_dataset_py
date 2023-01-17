import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.model_selection import train_test_split
df_test=pd.read_csv("../input/test.csv")

df_train=pd.read_csv("../input/train.csv")

df_train.head()
survived_train=df_train.Survived
data=pd.concat([df_train.drop(['Survived'],axis=1),df_test])
data.info()
data.Age=data.Age.fillna(data.Age.median())

data.Fare=data.Fare.fillna(data.Fare.median())
data.info()
data.head(10)
data['Sex']=data['Sex'].apply(lambda x: 1 if x=='male' else 0)
data.head()
data=data[['Sex','Fare','Age','Pclass','SibSp']]
data.head()
data_train=data.iloc[:891]

data_test=data.iloc[891:]
X=data_train.values
test=data_test.values
y=survived_train.values
clf=tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)
y_pred=clf.predict(test)
df_test['Survived']=y_pred
df_test.head(5)
pwd

df_test[['PassengerId','Survived']].to_csv("kaggle",index=False)