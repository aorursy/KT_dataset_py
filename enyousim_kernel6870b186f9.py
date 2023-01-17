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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df1=pd.read_csv("/kaggle/input/titanic/train.csv")
df1.head()
df1.drop(["Name","PassengerId","Ticket"],axis =1,inplace=True)
df1.head()
sns.heatmap(df1.isna())
df1.count()

df1.drop("Cabin",axis=1,inplace=True)
df1.head()
sns.boxplot("Pclass","Age",hue="Sex",data=df1)
def fillgap(dataframe):#3 terms 1st term is Age, second term is Pclass 3rd term is Sex

    Age=dataframe[0]

    Pclass =dataframe[1]

    Sex=dataframe[2]

    if pd.isnull(Age):

        if Pclass==1:#1st class

            if Sex=="male":

                return 40

            else:

                return 35

        elif Pclass==2:

            if Sex=="male":

                return 30

            else:

                    return 29

        elif Pclass==3:

            if Sex=="male":

                return 26

            else:

                return 21

    else:

        return Age
df1["Age"]=df1[["Age","Pclass","Sex"]].apply(fillgap,axis=1)
sns.heatmap(df1.isna())
df1.head()
sns.countplot("Survived",data=df1,hue="SibSp")
sns.countplot("Survived",data=df1,hue="Sex")
sns.countplot("Survived",data=df1,hue="Parch")
sns.countplot("Survived",data=df1,hue="Embarked")
sns.boxplot("Embarked","Fare",data=df1)
sex=pd.get_dummies(df1["Sex"],drop_first=True)

embarked=pd.get_dummies(df1["Embarked"],drop_first=True)
df1.drop(["Sex","Embarked","Fare"],axis=1,inplace=True)

df1.head()
df1.head()
df1=pd.concat([df1,sex,embarked],axis=1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

df1.head()
xtrain=df1.drop("Survived",axis=1)

ytrain=df1["Survived"]
logmodel.fit(xtrain,ytrain)
#First part done now to test model

df2=pd.read_csv("/kaggle/input/titanic/test.csv")
df2.head()
df2.drop(["Name","PassengerId","Ticket","Cabin","Fare"],axis =1,inplace=True)

#df2["Age"]=df2[["Age","Pclass","Sex"]].apply(fillgap,axis=1)

#sex2=pd.get_dummies(df2["Sex"],drop_first=True)

#embarked2=pd.get_dummies(df2["Embarked"],drop_first=True)

#df2.drop(["Sex","Embarked"],axis=1,inplace=True)

#df2=pd.concat([df2,sex,embarked],axis=1)
df2.head()
sns.heatmap(df2.isna())
df2["Age"]=df2[["Age","Pclass","Sex"]].apply(fillgap,axis=1)
sns.heatmap(df2.isna())
sex2=pd.get_dummies(df2["Sex"],drop_first=True)

embarked2=pd.get_dummies(df2["Embarked"],drop_first=True)

df2.drop(["Sex","Embarked"],axis=1,inplace=True)

df2=pd.concat([df2,sex2,embarked2],axis=1)
sns.heatmap(df2.isna())
df2.head()
xtrain=df2
pred=logmodel.predict(xtrain)#Fare has problem check
from sklearn.metrics import confusion_matrix,classification_report
ans=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(classification_report(pred,ans["Survived"]))

print(confusion_matrix(pred,ans["Survived"]))

result=pd.DataFrame(pred)



df3=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
result["PassengerID"]=df3["PassengerId"]
result.to_csv('result.csv')