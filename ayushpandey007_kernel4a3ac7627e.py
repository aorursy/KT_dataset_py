import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.metrics import accuracy_score



# Figures inline and set visualization style

%matplotlib inline

sns.set()
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
survived_train=df_train.Survived

data=pd.concat([df_train.drop(['Survived'],axis=1),df_test],sort=False)
data.info()
data['Age']=data.Age.fillna(data.Age.median())
data['Fare']=data.Fare.fillna(data.Fare.median())
data=pd.get_dummies(data,columns=['Sex'],drop_first=True)

data=data[['Sex_male','Fare','Pclass','Age','SibSp']]

data.head()
X=data.iloc[:891].values

y_test=data.iloc[891:].values

y=survived_train.values
clf=tree.DecisionTreeClassifier(max_depth=3)

clf.fit(X,y)
y_pred=clf.predict(y_test)

df_test['Survived']=y_pred

df_test[['PassengerId','Survived']].to_csv('my_submission.csv', index=False)