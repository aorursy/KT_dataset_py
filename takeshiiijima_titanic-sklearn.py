# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
baseDir='/kaggle/input/titanic/'
train= pd.read_csv(baseDir +"train.csv")

test= pd.read_csv(baseDir + "test.csv")

train.head(30)
train.info()
# replace

train = pd.read_csv(baseDir + "train.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)

test = pd.read_csv(baseDir + "test.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)



# replace

train["Age"].fillna(train.Age.mean(), inplace=True) 

train["Embarked"].fillna(train.Embarked.mean(), inplace=True)
combine1 = [train]



for train in combine1: 

    train['Salutation'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 

        

for train in combine1: 

        train['Salutation'] = train['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        train['Salutation'] = train['Salutation'].replace('Mlle', 'Miss')

        train['Salutation'] = train['Salutation'].replace('Ms', 'Miss')

        train['Salutation'] = train['Salutation'].replace('Mme', 'Mrs')

        del train['Name']

        

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 

for train in combine1: 

        train['Salutation'] = train['Salutation'].map(Salutation_mapping) 

        train['Salutation'] = train['Salutation'].fillna(0)
for train in combine1: 

        train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])

        train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x)) 

        train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'], np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 

        train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x)) 

        del train['Ticket'] 

        

train['Ticket_Lett'] = train['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)
for train in combine1: 

    train['Cabin_Lett'] = train['Cabin'].apply(lambda x: str(x)[0]) 

    train['Cabin_Lett'] = train['Cabin_Lett'].apply(lambda x: str(x)) 

    train['Cabin_Lett'] = np.where((train['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),train['Cabin_Lett'], np.where((train['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))

del train['Cabin'] 



train['Cabin_Lett']=train['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)
train.head(10)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1



for train in combine1:

    train['IsAlone'] = 0

    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
train.head(30)
train_data = train.values

xs = train_data[:, 2:] 

y  = train_data[:, 1] 

test.info()
test["Age"].fillna(train.Age.mean(), inplace=True)

test["Fare"].fillna(train.Fare.mean(), inplace=True)



combine = [test]



for test in combine:

    test['Salutation'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

for test in combine:

    test['Salutation'] = test['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    test['Salutation'] = test['Salutation'].replace('Mlle', 'Miss')

    test['Salutation'] = test['Salutation'].replace('Ms', 'Miss')

    test['Salutation'] = test['Salutation'].replace('Mme', 'Mrs')

    del test['Name']

    

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for test in combine:

    test['Salutation'] = test['Salutation'].map(Salutation_mapping)

    test['Salutation'] = test['Salutation'].fillna(0)



for test in combine:

        test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])

        test['Ticket_Lett'] = test['Ticket_Lett'].apply(lambda x: str(x))

        test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'],

                                   np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),'0', '0'))

        test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))

        del test['Ticket']

        

test['Ticket_Lett'] = test['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 



for test in combine:

        test['Cabin_Lett'] = test['Cabin'].apply(lambda x: str(x)[0])

        test['Cabin_Lett'] = test['Cabin_Lett'].apply(lambda x: str(x))

        test['Cabin_Lett'] = np.where((test['Cabin_Lett']).isin(['T', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']),test['Cabin_Lett'],

                                   np.where((test['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),'0','0'))        

        del test['Cabin']

        

test['Cabin_Lett'] = test['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1).replace("G",1) 

test["FamilySize"] = train["SibSp"] + train["Parch"] + 1



for test in combine:

    test['IsAlone'] = 0

    test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

    

test_data = test.values

xs_test = test_data[:, 1:]
from sklearn.ensemble import RandomForestClassifier



random_forest=RandomForestClassifier()

random_forest.fit(xs, y)

Y_pred = random_forest.predict(xs_test)

%matplotlib inline 

import matplotlib.pyplot as plt

import seaborn as sns



g = sns.factorplot(x="Sex", y="Survived",  data=train,size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
sns.countplot(x='Sex', data = train)
g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train, size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.factorplot(x="Salutation", y="Survived",  data=train, size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

del train['PassengerId']

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
sns.countplot(x='FamilySize', data = train, hue = 'Survived')