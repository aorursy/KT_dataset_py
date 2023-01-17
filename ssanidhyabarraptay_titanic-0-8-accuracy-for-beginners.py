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
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
#So now lets import our training and test datasets
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
train.info()
#Lets check the count of null values in our training dataset
train.isnull().sum()
#Assign y as survived column from training dataset
y=train['Survived']
#Cabin column is really intresting and can be quite usefull so we wont be deleting it and we'll replace the null values by any letter in
#this case consider U and we will fill the null values for both train and test data

train["Cabin"].unique()
train["Cabin"] = train["Cabin"].fillna('U')
train["Cabin"] = train["Cabin"].apply(lambda x: x[0])


test["Cabin"].unique()
test["Cabin"] = test["Cabin"].fillna('U')
test["Cabin"] = test["Cabin"].apply(lambda x: x[0])
#Lets make us new column Family size which will give us the total number of family size for every passenger  
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
#Name list contains the title which can be very usefull for survival prediction ,so we will extract the title from evey name and remove 
#the remaining data

train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train['Name'].unique()

test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles1 = test['Name'].unique()
titles,titles1
#Name Column
title_train=pd.get_dummies(train['Name'],drop_first=True,prefix='Title')
train=pd.concat([train,title_train],axis=1)
train.drop(columns=['Name'],axis=1,inplace=True)

title_test=pd.get_dummies(test['Name'],drop_first=True,prefix='Title')
test=pd.concat([test,title_test],axis=1)
test.drop(columns=['Name'],axis=1,inplace=True)
#Sibsp Column
Sib_train=pd.get_dummies(train['SibSp'],drop_first=True,prefix='Sib')
train=pd.concat([train,Sib_train],axis=1)
train.drop(columns=['SibSp'],axis=1,inplace=True)

Sib_test=pd.get_dummies(test['SibSp'],drop_first=True,prefix='Sib')
test=pd.concat([test,Sib_test],axis=1)
test.drop(columns=['SibSp'],axis=1,inplace=True)
#Cabin Column
cabin_train=pd.get_dummies(train['Cabin'],drop_first=True,prefix='Cabin')
train=pd.concat([train,cabin_train],axis=1)
train.drop(columns=['Cabin'],axis=1,inplace=True)

cabin_test=pd.get_dummies(test['Cabin'],drop_first=True,prefix='Cabin')
test=pd.concat([test,cabin_test],axis=1)
test.drop(columns=['Cabin'],axis=1,inplace=True)
#Parch Column
parch_train=pd.get_dummies(train['Parch'],drop_first=True,prefix='Parch')
train=pd.concat([train,parch_train],axis=1)
train.drop(columns=['Parch'],axis=1,inplace=True)

parch_test=pd.get_dummies(test['Parch'],drop_first=True,prefix='Parch')
test=pd.concat([test,parch_test],axis=1)
test.drop(columns=['Parch'],axis=1,inplace=True)
train.Age.min(),train.Age.max()
#As we can see above the value of Age column ranges from 0.42 to 80.0.So we will create a new column Age Attributes that will 
#define the attribute of person by their age
#1) For training data:-
condition_train=[
    (train.Age<10),
    (train.Age<20),
    (train.Age<50),
    (train.Age<80)
    
]

Age_attributes_train=['minor','teenage','Adult','Old']

train['Age_attributes']=np.select(condition_train,Age_attributes_train,default='Ages')

#2) For testing data:-
condition_test=[
    (test.Age<10),
    (test.Age<20),
    (test.Age<50),
    (test.Age<80)
    
]
Age_attributes_test=['minor','teenage','Adult','Old']

test['Age_attributes']=np.select(condition_test,Age_attributes_test,default='Ages')

#Now we will drop the age column as we've created a new column of Age Attributes
train.drop(columns=['Age'],axis=1,inplace=True)
test.drop(columns=['Age'],axis=1,inplace=True)
train.isnull().sum(),   test.isnull().sum()
#Feature Encoding Continues:
Age_train=pd.get_dummies(train['Age_attributes'],drop_first=True,prefix='Ages_of_people')
Age_test=pd.get_dummies(test['Age_attributes'],drop_first=True,prefix='Ages_of_people')

train=pd.concat([train,Age_train],axis=1)
test=pd.concat([test,Age_test],axis=1)

pdclass_train=pd.get_dummies(train['Pclass'],drop_first=True,prefix='Class')
pdclass_test=pd.get_dummies(test['Pclass'],drop_first=True,prefix='Class')
train=pd.concat([train,pdclass_train],axis=1)
test=pd.concat([test,pdclass_test],axis=1)
train.drop(columns=['Pclass'],axis=1,inplace=True)
test.drop(columns=['Pclass'],axis=1,inplace=True)
sex=pd.get_dummies(train['Sex'],drop_first=True,prefix='Sex')
sex1=pd.get_dummies(test['Sex'],drop_first=True,prefix='Sex')
Embarked=pd.get_dummies(train['Embarked'],drop_first=True,prefix='Embarked')
Embarked1=pd.get_dummies(test['Embarked'],drop_first=True,prefix='Embarked')
test=pd.concat([test,sex1,Embarked1],axis=1)
train=pd.concat([train,sex,Embarked],axis=1)
train.drop(columns=['Sex','Embarked'],axis=1,inplace=True)  
test.drop(columns=['Sex','Embarked'],axis=1,inplace=True)
train.isnull().sum()
#We'll drop the age attributes now as we've created dummy variables for it
train.drop(columns=['Age_attributes'],axis=1,inplace=True)
test.drop(columns=['Age_attributes'],axis=1,inplace=True)
test.isnull().sum()
#Firstly we'll have to drop Parch_9 column from test dataset as this column doesnt exist in training data
test.drop(columns=['Parch_9'],axis=1,inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
#We wont be needing Ticket column as it has lot of unique values and feature encoding wont be possible, so we'll just dropit
train.drop(columns=['Ticket'],axis=1,inplace=True)
test.drop(columns=['Ticket'],axis=1,inplace=True)
#From the title column we've created we have a lot of new colums from train set that aren' present in test set
#so we will delete those columns from our train dataset
train.drop(columns=['Title_Mlle','Title_Col','Title_the Countess','Title_Jonkheer','Title_Major','Title_Lady','Title_Mlle','Title_Mme','Title_Sir'],inplace=True,axis=1)
#Cabin_T is also an extra colun in train set which isnt present in test set so we will drop that as well
train.drop(columns=['Cabin_T'],axis=1,inplace=True)
test.rename(columns={'Title_Dona':'Title_Don'},inplace=True)
train.drop(columns=['Survived'],axis=1,inplace=True)
#Alright our both datasets are ready
#We'll just take a look at their shape
train.shape,test.shape
#This is he model that worked best for me and got me good accuracy
#You can use it as well  :)
leaderboard_model = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=None,
                                           n_jobs=-1,
                                           verbose=1)
leaderboard_model.fit(train,y)
leaderboard_model.score(train,y)
#0.85 now thats a great accuracy for a begineer
#Make sure to carefully follow the steps throghout to get this accuracy
y_pred=leaderboard_model.predict(test)
df_gendercsv=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
df_gendercsv.to_csv('gender.csv',index=False)
#Thats it guys thanks for sticking throghout
#Bye for now
