# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



print("Setup Complete")

# Any results you write to the current directory are saved as output.
# Reading data 

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')



#Defining Final DF

submission = pd.DataFrame(0, index=np.arange(0, len(test)), columns=['PassengerId','Survived'])

k_fold = KFold(n_splits=15, shuffle=True, random_state=0)
#Survival Based upon Sex

Sex = pd.crosstab(train['Sex'],train['Survived'])

Sex.plot(kind="bar",stacked=True,title="Suvival based on sex")

#Suvival based on Pclass

Pclass = pd.crosstab(train['Pclass'],train['Survived'])

Pclass.plot(kind="bar",stacked=True,title="Suvival based on Pclass")

#Feature Logics



#Age 

def CalculateAge(dataframe):

    dataframe['Title']=dataframe.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

    normalized_titles = {"Capt":"Officer","Col":"Officer","Major":"Officer","Jonkheer":"Royalty","Don":"Royalty",

    "Sir" :"Royalty","Dr":"Officer","Rev":"Officer","the Countess":"Royalty","Dona":"Royalty","Mme":"Mrs",

    "Mlle":"Miss","Ms":"Mrs","Mr" :"Mr","Mrs" :"Mrs","Miss" :"Miss","Master" :"Master","Lady" :"Royalty"}

    dataframe.Title = dataframe.Title.map(normalized_titles)

    grouped = dataframe.groupby(['Sex','Pclass','Title'])

    dataframe.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

    grouped.Age.median()

    return (dataframe)



#Famil Size calculation

def CalclateFamilySize(dataframe):

    dataframe['FamilySize']=dataframe['SibSp']+dataframe['Parch']+1

    dataframe['Embarked'].fillna(value='S',inplace=True)

    return(dataframe)





train = CalculateAge(train)

train = CalclateFamilySize(train)



test = CalculateAge(test)

test = CalclateFamilySize(test)



Features = ['Pclass','Sex','Age','FamilySize','Title']



train_pred = train[Features]

test_pred = test[Features]



train_pred = pd.get_dummies(train_pred)

test_pred = pd.get_dummies(test_pred)
#Assigning Variables for model

X=train_pred

y=train.Survived
#Splitting data 

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#random forest classifier

model = RandomForestClassifier(n_estimators = 270)

model.fit(x_train,y_train)

pred=model.predict(x_test)

A = accuracy_score(y_test, pred)*100

print('Accuracy %f ' %(A))


pred_test = model.predict(test_pred)



submission['Survived']=pred_test

submission['PassengerId']=test['PassengerId']



submission['Survived'].replace(0, 'Dead',inplace=True) 

submission['Survived'].replace(1, 'Survived',inplace=True)



print(submission)


