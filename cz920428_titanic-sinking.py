# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt 

train = pd.read_csv('../input/train.csv')

#Check training dataset

#12 columns. PassengerID(unique and incremental from 1). 

#Survived 0,1   prediction value

#Pclass 1,2,3  feature 1: better survival chance 

#Name passenger name. It shouldn't be feature as it does not decide survival rate 

#Sex male female Male: better survival chance 

#Age nan, integer, float 

#SibSp 0 1 2 3 4 5 8 0: better survival chance 

#Parch 0 1 2 3 4 5 6 0: better survival chance 

#Ticket ticket number It shouldn't be feature as it does not decide survival rate 

#Fare ticket fare It shouldn't be feature as it does not decide survival rate 

#Cabin nan, need to check different starting letter. Lots of missing data 

#Embarked S C Q nan C

train.head()

train.Survived.unique()

train.Pclass.unique()

train.Sex.unique()

train.Age.unique()

train.SibSp.unique()

train.Parch.unique()

train.Cabin.unique()

train.Embarked.unique()

pclass = train.groupby(['Pclass','Survived']).PassengerId.count().unstack()

pclass.plot.bar()

sex = train.groupby(['Sex','Survived']).PassengerId.count().unstack()

sex.plot.bar()

SibSp = train.groupby(['SibSp','Survived']).PassengerId.count().unstack()

SibSp.plot.bar()

Parch = train.groupby(['Parch','Survived']).PassengerId.count().unstack()

Parch.plot.bar()

Embarked = train.groupby(['Embarked','Survived']).PassengerId.count().unstack()

Embarked.plot.bar()

AgeGroup = train.groupby([pd.cut(train.Age,np.arange(0,max(train.Age)+10,10)),'Survived']).PassengerId.count().unstack()

AgeGroup.plot.bar()

FareGroup = train.groupby([pd.cut(train.Fare,np.arange(0,max(train.Fare)+50,50)),'Survived']).PassengerId.count().unstack()

FareGroup.plot.bar()

CabinGroup = train.groupby([train.Cabin.str[:1],'Survived']).PassengerId.count().unstack()

CabinGroup.plot.bar()

FareCabin = train.groupby([train.Cabin.str[:1],pd.cut(train.Fare,np.arange(0,max(train.Fare)+50,50))]).PassengerId.count().unstack()

FareCabin



import random as rand

import math

#Total record of training data: 891 

len(train)

#Check number os NaN values in each column 

train.isnull().sum()

#Age: 177 null value 

#Cabin: 687 null value too many null values. ignore this feature 

#Embarked: 2 null value 

##Age mean: 29.6991764705882; sd: 14.526497332334044

train.Age.mean()

train.Age.std()

##Create the age range with mean +/- sd 

np.arange(train.Age.mean()-train.Age.std(), train.Age.mean()+train.Age.std())

##Randomly assign the age to the Nan values 

train['AgeModified'] = train['Age']

math.isnan(train.iloc[0]['Age'])

for i in range(len(train)):

    if math.isnan(train.iloc[i]['Age']) == True:

       train.loc[i, 'AgeModified'] = rand.uniform(train.Age.mean()-train.Age.std(), train.Age.mean()+train.Age.std())

##Most frequent: S 

train.groupby('Embarked').PassengerId.count()  

train['EmbarkedModified'] = train['Embarked']

for i in range(len(train)):

    if pd.isnull(train.loc[i,'Embarked']) == True: 

        train.loc[i,'EmbarkedModified'] = 'S'
#check AgeModified column 

train[train['Age'].isnull()].head()
train[train['Embarked'].isnull()]
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

for i in range(len(train)):

    if train.loc[i,'SibSp'] == 0 and train.loc[i,'Parch'] == 0:

        train.loc[i,'isAlone'] = 1 

    else: 

        train.loc[i,'isAlone'] = 0

train['isAlone'] = train['isAlone'].astype(int)
train.head()
#Age 

train.AgeCategorical = train.AgeModified

train.loc[train['AgeModified']<=10,'AgeCategorical'] = 0 

train.loc[(train['AgeModified']>10) & (train['AgeModified']<=20),'AgeCategorical'] = 1 

train.loc[(train['AgeModified']>20) & (train['AgeModified']<=30),'AgeCategorical'] = 2

train.loc[(train['AgeModified']>30) & (train['AgeModified']<=40),'AgeCategorical'] = 3 

train.loc[(train['AgeModified']>40) & (train['AgeModified']<=50),'AgeCategorical'] = 4 

train.loc[(train['AgeModified']>50) & (train['AgeModified']<=60),'AgeCategorical'] = 5

train.loc[(train['AgeModified']>60) & (train['AgeModified']<=70),'AgeCategorical'] = 6

train.loc[(train['AgeModified']>70) & (train['AgeModified']<=80),'AgeCategorical'] = 7

train['AgeCategorical'] = train['AgeCategorical'].astype(int)

train.head()

#Sex 

train.SexCategorical = train.Sex

train.loc[train['Sex']=='male','SexCategorical'] = 0 

train.loc[train['Sex']=='female','SexCategorical'] = 1

train['SexCategorical'] = train['SexCategorical'].astype(int)

train.head()

#Embarked 

train.EmbarkedCategorical = train.EmbarkedModified

train.loc[train['EmbarkedModified'] == 'S', 'EmbarkedCategorical'] = 0

train.loc[train['EmbarkedModified'] == 'C', 'EmbarkedCategorical'] = 1

train.loc[train['EmbarkedModified'] == 'Q', 'EmbarkedCategorical'] = 2

train['EmbarkedCategorical'] = train['EmbarkedCategorical'].astype(int)

train.head()

#Fare 

train.FareCategorical = train.Fare

train.loc[train['Fare']<=50,'FareCategorical'] = 0 

train.loc[(train['Fare']>50) & (train['Fare']<=100),'FareCategorical'] = 1 

train.loc[(train['Fare']>100) & (train['Fare']<=150),'FareCategorical'] = 2

train.loc[(train['Fare']>150) & (train['Fare']<=200),'FareCategorical'] = 3 

train.loc[(train['Fare']>200) & (train['Fare']<=250),'FareCategorical'] = 4 

train.loc[(train['Fare']>250) & (train['Fare']<=300),'FareCategorical'] = 5

train.loc[(train['Fare']>300) & (train['Fare']<=350),'FareCategorical'] = 6

train.loc[(train['Fare']>350) & (train['Fare']<=400),'FareCategorical'] = 7

train.loc[(train['Fare']>400) & (train['Fare']<=450),'FareCategorical'] = 8

train.loc[(train['Fare']>450) & (train['Fare']<=500),'FareCategorical'] = 9

train.loc[(train['Fare']>500) & (train['Fare']<=550),'FareCategorical'] = 10

train['FareCategorical'] = train['FareCategorical'].astype(int)

train.head()
from sklearn.ensemble import RandomForestClassifier

features = ['Pclass','FamilySize','isAlone','AgeCategorical','SexCategorical',

                        'EmbarkedCategorical','FareCategorical','SibSp','Parch']

output = train['Survived']

clf = RandomForestClassifier(n_jobs=-1,n_estimators=500)

clf.fit(train[features], output)

train['prediction'] = clf.predict(train[features])

train.head()

##accuracy on training data: 88.33%

train[train['Survived']==train['prediction']].PassengerId.count()/float(train.PassengerId.count())

#feature_importance = pd.DataFrame({'features':train[features].columns.values, 'importance':clf.feature_importances_})

#feature_importance.sort_values('importance',ascending=False)

test = pd.read_csv('../input/test.csv')

test['AgeModified'] = test['Age']

math.isnan(test.iloc[0]['Age'])

for i in range(len(test)):

    if math.isnan(test.iloc[i]['Age']) == True:

       test.loc[i, 'AgeModified'] = rand.uniform(test.Age.mean()-test.Age.std(), test.Age.mean()+test.Age.std())

##Most frequent: S 

test.groupby('Embarked').PassengerId.count()  

test['EmbarkedModified'] = test['Embarked']

for i in range(len(test)):

    if pd.isnull(test.loc[i,'Embarked']) == True: 

        test.loc[i,'EmbarkedModified'] = 'S'

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

for i in range(len(test)):

    if test.loc[i,'SibSp'] == 0 and test.loc[i,'Parch'] == 0:

        test.loc[i,'isAlone'] = 1 

    else: 

        test.loc[i,'isAlone'] = 0

test['isAlone'] = test['isAlone'].astype(int)

#Age 

test.AgeCategorical = test.AgeModified

test.loc[test['AgeModified']<=10,'AgeCategorical'] = 0 

test.loc[(test['AgeModified']>10) & (test['AgeModified']<=20),'AgeCategorical'] = 1 

test.loc[(test['AgeModified']>20) & (test['AgeModified']<=30),'AgeCategorical'] = 2

test.loc[(test['AgeModified']>30) & (test['AgeModified']<=40),'AgeCategorical'] = 3 

test.loc[(test['AgeModified']>40) & (test['AgeModified']<=50),'AgeCategorical'] = 4 

test.loc[(test['AgeModified']>50) & (test['AgeModified']<=60),'AgeCategorical'] = 5

test.loc[(test['AgeModified']>60) & (test['AgeModified']<=70),'AgeCategorical'] = 6

test.loc[(test['AgeModified']>70) & (test['AgeModified']<=80),'AgeCategorical'] = 7

test['AgeCategorical'] = test['AgeCategorical'].astype(int)

test.head()
#Sex 

test.SexCategorical = test.Sex

test.loc[test['Sex']=='male','SexCategorical'] = 0 

test.loc[test['Sex']=='female','SexCategorical'] = 1

test['SexCategorical'] = test['SexCategorical'].astype(int)

test.head()

#Embarked 

test.EmbarkedCategorical = test.EmbarkedModified

test.loc[test['EmbarkedModified'] == 'S', 'EmbarkedCategorical'] = 0

test.loc[test['EmbarkedModified'] == 'C', 'EmbarkedCategorical'] = 1

test.loc[test['EmbarkedModified'] == 'Q', 'EmbarkedCategorical'] = 2

test['EmbarkedCategorical'] = test['EmbarkedCategorical'].astype(int)

test.head()
#Fare 1 null value for testing dataset

test.FareCategorical = test.Fare

for i in range(len(test)):

    if pd.isnull(test.loc[i,'Fare']) == True: 

        test.loc[i,'FareCategorical'] = test['Fare'].mean()
test.loc[test['Fare']<=50,'FareCategorical'] = 0 

test.loc[(test['Fare']>50) & (test['Fare']<=100),'FareCategorical'] = 1 

test.loc[(test['Fare']>100) & (test['Fare']<=150),'FareCategorical'] = 2

test.loc[(test['Fare']>150) & (test['Fare']<=200),'FareCategorical'] = 3 

test.loc[(test['Fare']>200) & (test['Fare']<=250),'FareCategorical'] = 4 

test.loc[(test['Fare']>250) & (test['Fare']<=300),'FareCategorical'] = 5

test.loc[(test['Fare']>300) & (test['Fare']<=350),'FareCategorical'] = 6

test.loc[(test['Fare']>350) & (test['Fare']<=400),'FareCategorical'] = 7

test.loc[(test['Fare']>400) & (test['Fare']<=450),'FareCategorical'] = 8

test.loc[(test['Fare']>450) & (test['Fare']<=500),'FareCategorical'] = 9

test.loc[(test['Fare']>500) & (test['Fare']<=550),'FareCategorical'] = 10

test['FareCategorical'] = test['FareCategorical'].astype(int)

test.head()
test['Survived'] = clf.predict(test[features])

test.head()

test.groupby('Survived').PassengerId.count()

test.to_csv('out.csv',columns=['PassengerId','Survived'],index=False)