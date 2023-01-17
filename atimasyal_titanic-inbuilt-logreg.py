import pandas as pd

import math

import numpy as np

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
def f(s):

    if s=="male":

        return 1

    else:

        return 2
def f1(s):

    if s=="S":

        return 1

    elif s=="Q":

        return 2

    else:

        return 3
df=pd.read_csv("../input/titanic_train.csv")

del df["Name"]

del df["Ticket"]

del df["Cabin"]

#del df["Embarked"] # I think this feature is important to decide the output

del df["SibSp"]

del df["Parch"]

amale=df[df.Sex=='male']

afemale=df[df.Sex=='female']

amale.Age.mean(), afemale.Age.mean()

df.loc[(df['Sex'] == 'male') & (df['Age'].isnull()) , 'Age'] = amale.Age.mean()

df.loc[(df['Sex'] == 'female') & (df['Age'].isnull()) , 'Age'] = afemale.Age.mean()
df[0:20]
df["Sex"]=df.Sex.apply(f)

df["Embarked"]=df.Embarked.apply(f1)

#df["Parch"]=df["Parch"]+df["SibSp"]

#df.loc[(df["Parch"]>0)]=1

#df["Parch"]=df["Parch"]+1

#del df["SibSp"]

data=df.values

N= len(data[0])

M = len(data)

x=data[:, 0:N-1]

y=data[:, N-1]

scaler=preprocessing.MinMaxScaler(feature_range=(1, 3))

scaler.fit(x)

x=scaler.transform(x)

clf=LogisticRegression(max_iter=1000)

clf.fit(x, y)
clf.score(x, y)
df1=pd.read_csv("../input/titanic_test.csv")

del df1["Name"]

del df1["Ticket"]

del df1["Cabin"]

#del df1["Embarked"]

del df1["SibSp"]

del df1["Parch"]

amale=df1[df1.Sex=='male']

afemale=df1[df1.Sex=='female']

df1.loc[(df1['Sex'] == 'male') & (df1['Age'].isnull()) , 'Age'] = amale.Age.mean()

df1.loc[(df1['Sex'] == 'female') & (df1['Age'].isnull()) , 'Age'] = afemale.Age.mean()

df1["Sex"]=df1.Sex.apply(f)

df1["Embarked"]=df1.Embarked.apply(f1)

#df1.loc[(df1["Parch"]>0)]=1

#df1["Parch"]=df1["Parch"]+1

#del df1["SibSp"]

test=df1.values

test=scaler.transform(test)
ypre=clf.predict(test)
np.savetxt(X=ypre,fname="titanic_pred.csv")