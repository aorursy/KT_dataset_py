# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from scipy.stats import mode

from sklearn.svm import SVC

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.ensemble import AdaBoostClassifier
titanic=pd.read_csv("/kaggle/input/train.csv")

df=titanic.copy()

df.head()
df.isnull().sum()
test=pd.read_csv("/kaggle/input/test.csv")

test_df=test.copy()

test_df.isnull().sum()
df["Initial"]=df["Name"].str.extract('([A-Za-z]+)\.')

print(df["Initial"].unique())

df["Initial"].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                      ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
df.groupby("Initial")["Age"].mean()
df.loc[(df.Age.isnull())&(df.Initial=='Master'),"Age"]=5

df.loc[(df.Age.isnull())&(df.Initial=='Miss'),"Age"]=22

df.loc[(df.Age.isnull())&(df.Initial=='Mr'),"Age"]=33

df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),"Age"]=36

df.loc[(df.Age.isnull())&(df.Initial=='Other'),"Age"]=46
print(df.Embarked.mode())

df.Embarked.fillna("S",inplace=True)
df["Family_size"]=df["SibSp"]+df["Parch"]

df["Alone"]=0

df.loc[df.Family_size==0,"Alone"]=1
df['Age_band']=0

df.loc[df['Age']<=16,'Age_band']=0

df.loc[(df['Age']>16)&(df['Age']<=32),'Age_band']=1

df.loc[(df['Age']>32)&(df['Age']<=48),'Age_band']=2

df.loc[(df['Age']>48)&(df['Age']<=64),'Age_band']=3

df.loc[df['Age']>64,'Age_band']=4
df['Fare_Range']=pd.qcut(df['Fare'],4)

df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
df['Fare_cat']=0

df.loc[df['Fare']<=7.91,'Fare_cat']=0

df.loc[(df['Fare']>7.91)&(df['Fare']<=14.454),'Fare_cat']=1

df.loc[(df['Fare']>14.454)&(df['Fare']<=31),'Fare_cat']=2

df.loc[(df['Fare']>31)&(df['Fare']<=513),'Fare_cat']=3
df['Sex'].replace(['male','female'],[0,1],inplace=True)

df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
df.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
df.describe()
df.isnull().sum()
test_df["Initial"]=test_df["Name"].str.extract('([A-Za-z]+)\.')

print(test_df["Initial"].unique())

test_df["Initial"].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Dona'],

                      ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Master'),"Age"]=5

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Miss'),"Age"]=22

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mr'),"Age"]=33

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mrs'),"Age"]=36

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Other'),"Age"]=46

test_df.Embarked.fillna("S",inplace=True)
test_df["Family_size"]=test_df["SibSp"]+test_df["Parch"]

test_df["Alone"]=0

test_df.loc[df.Family_size==0,"Alone"]=1
test_df['Age_band']=0

test_df.loc[test_df['Age']<=16,'Age_band']=0

test_df.loc[(test_df['Age']>16)&(test_df['Age']<=32),'Age_band']=1

test_df.loc[(test_df['Age']>32)&(test_df['Age']<=48),'Age_band']=2

test_df.loc[(test_df['Age']>48)&(test_df['Age']<=64),'Age_band']=3

test_df.loc[test_df['Age']>64,'Age_band']=4
test_df[test_df.Fare.isnull()]
test_df.loc[(test_df.Pclass==3) & (test_df.Embarked=="S"),"Fare"].mean()

test_df.Fare.fillna(14,inplace=True)
test_df['Fare_cat']=0

test_df.loc[test_df['Fare']<=7.91,'Fare_cat']=0

test_df.loc[(test_df['Fare']>7.91)&(test_df['Fare']<=14.454),'Fare_cat']=1

test_df.loc[(test_df['Fare']>14.454)&(test_df['Fare']<=31),'Fare_cat']=2

test_df.loc[(test_df['Fare']>31)&(test_df['Fare']<=513),'Fare_cat']=3
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

test_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

test_df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
test_df.drop(['Name','Age','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)
test_df.describe(include="all")
test_df.isnull().sum()
ytrain=df["Survived"]

del df["Survived"]
xtrain=df.values

xtest=test_df.values
xtrain.shape,ytrain.shape,xtest.shape
n_estimators=list(range(100,1100,100))

learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}

gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)

gd.fit(xtrain,ytrain)

print(gd.best_score_)

print(gd.best_estimator_)
clf=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)

clf.fit(xtrain,ytrain)

print(clf.score(xtrain,ytrain))

ypred=clf.predict(xtest)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": ypred

    })

submission.to_csv('submission.csv', index=False)