from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import Imputer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import svm

from sklearn import tree

import xgboost as xgb

from sklearn.ensemble import BaggingRegressor

import numpy as np 

import pandas as pd 

from sklearn.metrics import roc_auc_score

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier

import seaborn as sns

from sklearn.metrics import accuracy_score
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submssion = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
train['Title'] = train.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

test['Title'] = test.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

train["Fsize"] = train["SibSp"] + train["Parch"] + 1

test["Fsize"] = test["SibSp"] + test["Parch"] + 1
train
train_df = train.drop('Survived',axis = 1)
data = pd.concat([train_df,test])
train = train.fillna(train.mean())

test = test.fillna(test.mean())
for name in train.columns:

    x = train[name].isna().sum()

    if x > 0:

        val_list = np.random.choice(train.groupby(name).count().index, x, p=train.groupby(name).count()['PassengerId'].values /sum(train.groupby(name).count()['PassengerId'].values))

        train.loc[train[name].isna(), name] = val_list
for name in test.columns:

    x = test[name].isna().sum()

    if x > 0:

        val_list = np.random.choice(test.groupby(name).count().index, x, p=test.groupby(name).count()['PassengerId'].values /sum(test.groupby(name).count()['PassengerId'].values))

        test.loc[test[name].isna(), name] = val_list
le = preprocessing.LabelEncoder()

for name in data.columns:

    if data[name].dtypes == "O":

        print(name)

        data[name] = data[name].astype(str)

        train[name] = train[name].astype(str)

        test[name] = test[name].astype(str)

        le.fit(data[name])

        train[name] = le.transform(train[name])

        test[name] = le.transform(test[name])
train.groupby(['Fsize','Survived']).count()
train.loc[train['Age'] < 10,'Age'] = 1

train.loc[(train['Age'] >= 10) & (train['Age'] < 20), 'Age' ] = 2

train.loc[(train['Age'] >= 20) & (train['Age'] < 30), 'Age' ] = 3

train.loc[(train['Age'] >= 30) & (train['Age'] < 40),'Age'] = 4

train.loc[(train['Age'] >= 40) & (train['Age'] < 50), 'Age' ] = 5

train.loc[(train['Age'] >= 50) & (train['Age'] < 60), 'Age' ] = 6

train.loc[(train['Age'] >= 60), 'Age' ] = 7



test.loc[test['Age'] < 10,'Age'] = 1

test.loc[(test['Age'] >= 10) & (test['Age'] < 20), 'Age' ] = 2

test.loc[(test['Age'] >= 20) & (test['Age'] < 30), 'Age' ] = 3

test.loc[(test['Age'] >= 30) & (test['Age'] < 40),'Age'] = 4

test.loc[(test['Age'] >= 40) & (test['Age'] < 50), 'Age' ] = 5

test.loc[(test['Age'] >= 50) & (test['Age'] < 60), 'Age' ] = 6

test.loc[(test['Age'] >= 60), 'Age' ] = 7







train.loc[train['Ticket'] < 100,'Ticket'] = 1

train.loc[(train['Ticket'] >= 100) & (train['Ticket'] < 200), 'Ticket' ] = 2

train.loc[(train['Ticket'] >= 200) & (train['Ticket'] < 300), 'Ticket' ] = 3

train.loc[(train['Ticket'] >= 300) & (train['Ticket'] < 400),'Ticket'] = 4

train.loc[(train['Ticket'] >= 400) & (train['Ticket'] < 500), 'Ticket' ] = 5

train.loc[(train['Ticket'] >= 500) & (train['Ticket'] < 600), 'Ticket' ] = 6

train.loc[(train['Ticket'] >= 600) & (train['Ticket'] < 700), 'Ticket' ] = 7

train.loc[(train['Ticket'] >= 700) & (train['Ticket'] < 800),'Ticket'] = 8

train.loc[(train['Ticket'] >= 800) & (train['Ticket'] < 900), 'Ticket' ] = 9

train.loc[(train['Ticket'] >= 900) & (train['Ticket'] < 1000), 'Ticket' ] = 10

train.loc[(train['Ticket'] >= 1000), 'Ticket' ] = 11





test.loc[test['Ticket'] < 100,'Ticket'] = 1

test.loc[(test['Ticket'] >= 100) & (test['Ticket'] < 200), 'Ticket' ] = 2

test.loc[(test['Ticket'] >= 200) & (test['Ticket'] < 300), 'Ticket' ] = 3

test.loc[(test['Ticket'] >= 300) & (test['Ticket'] < 400),'Ticket'] = 4

test.loc[(test['Ticket'] >= 400) & (test['Ticket'] < 500), 'Ticket' ] = 5

test.loc[(test['Ticket'] >= 500) & (test['Ticket'] < 600), 'Ticket' ] = 6

test.loc[(test['Ticket'] >= 600) & (test['Ticket'] < 700), 'Ticket' ] = 7

test.loc[(test['Ticket'] >= 700) & (test['Ticket'] < 800),'Ticket'] = 8

test.loc[(test['Ticket'] >= 800) & (test['Ticket'] < 900), 'Ticket' ] = 9

test.loc[(test['Ticket'] >= 900) & (test['Ticket'] < 1000), 'Ticket' ] = 10

test.loc[(test['Ticket'] >= 1000), 'Ticket' ] = 11



train.loc[train['Fare'] < 100,'Fare'] = 1

train.loc[(train['Fare'] >= 100) & (train['Fare'] < 200), 'Fare' ] = 2

train.loc[(train['Fare'] >= 200) & (train['Fare'] < 300), 'Fare' ] = 3

train.loc[(train['Fare'] >= 300) & (train['Fare'] < 400),'Fare'] = 4

train.loc[(train['Fare'] >= 400) & (train['Fare'] < 500), 'Fare' ] = 5

train.loc[(train['Fare'] >= 500) & (train['Fare'] < 600), 'Fare' ] = 6

train.loc[(train['Fare'] >= 600) & (train['Fare'] < 700), 'Fare' ] = 7

train.loc[(train['Fare'] >= 700) & (train['Fare'] < 800),'Fare'] = 8

train.loc[(train['Fare'] >= 800) & (train['Fare'] < 900), 'Fare' ] = 9

train.loc[(train['Fare'] >= 900) & (train['Fare'] < 1000), 'Fare' ] = 10

train.loc[(train['Fare'] >= 1000), 'Fare' ] = 11



test.loc[test['Fare'] < 100,'Fare'] = 1

test.loc[(test['Fare'] >= 100) & (test['Fare'] < 200), 'Fare' ] = 2

test.loc[(test['Fare'] >= 200) & (test['Fare'] < 300), 'Fare' ] = 3

test.loc[(test['Fare'] >= 300) & (test['Fare'] < 400),'Fare'] = 4

test.loc[(test['Fare'] >= 400) & (test['Fare'] < 500), 'Fare' ] = 5

test.loc[(test['Fare'] >= 500) & (test['Fare'] < 600), 'Fare' ] = 6

test.loc[(test['Fare'] >= 600) & (test['Fare'] < 700), 'Fare' ] = 7

test.loc[(test['Fare'] >= 700) & (test['Fare'] < 800),'Fare'] = 8

test.loc[(test['Fare'] >= 800) & (test['Fare'] < 900), 'Fare' ] = 9

test.loc[(test['Fare'] >= 900) & (test['Fare'] < 1000), 'Fare' ] = 10

test.loc[(test['Fare'] >= 1000), 'Fare' ] = 11





train.loc[train['Name'] < 100,'Name'] = 1

train.loc[(train['Name'] >= 100) & (train['Name'] < 200), 'Name' ] = 2

train.loc[(train['Name'] >= 200) & (train['Name'] < 300), 'Name' ] = 3

train.loc[(train['Name'] >= 300) & (train['Name'] < 400),'Name'] = 4

train.loc[(train['Name'] >= 400) & (train['Name'] < 500), 'Name' ] = 5

train.loc[(train['Name'] >= 500) & (train['Name'] < 600), 'Name' ] = 6

train.loc[(train['Name'] >= 600) & (train['Name'] < 700), 'Name' ] = 7

train.loc[(train['Name'] >= 700) & (train['Name'] < 800),'Name'] = 8

train.loc[(train['Name'] >= 800) & (train['Name'] < 900), 'Name' ] = 9

train.loc[(train['Name'] >= 900) & (train['Name'] < 1000), 'Name' ] = 10

train.loc[(train['Name'] >= 1000), 'Name' ] = 11



test.loc[test['Name'] < 100,'Name'] = 1

test.loc[(test['Name'] >= 100) & (test['Name'] < 200), 'Name' ] = 2

test.loc[(test['Name'] >= 200) & (test['Name'] < 300), 'Name' ] = 3

test.loc[(test['Name'] >= 300) & (test['Name'] < 400),'Name'] = 4

test.loc[(test['Name'] >= 400) & (test['Name'] < 500), 'Name' ] = 5

test.loc[(test['Name'] >= 500) & (test['Name'] < 600), 'Name' ] = 6

test.loc[(test['Name'] >= 600) & (test['Name'] < 700), 'Name' ] = 7

test.loc[(test['Name'] >= 700) & (test['Name'] < 800),'Name'] = 8

test.loc[(test['Name'] >= 800) & (test['Name'] < 900), 'Name' ] = 9

test.loc[(test['Name'] >= 900) & (test['Name'] < 1000), 'Name' ] = 10

test.loc[(test['Name'] >= 1000), 'Name' ] = 11



train.loc[train['Cabin'] < 100,'Cabin'] = 1

train.loc[(train['Cabin'] >= 100) & (train['Cabin'] < 200), 'Cabin' ] = 2

train.loc[(train['Cabin'] >= 200) & (train['Cabin'] < 300), 'Cabin' ] = 3

train.loc[(train['Cabin'] >= 300) & (train['Cabin'] < 400),'Cabin'] = 4

train.loc[(train['Cabin'] >= 400) & (train['Cabin'] < 500), 'Cabin' ] = 5

train.loc[(train['Cabin'] >= 500) & (train['Cabin'] < 600), 'Cabin' ] = 6

train.loc[(train['Cabin'] >= 600) & (train['Cabin'] < 700), 'Cabin' ] = 7

train.loc[(train['Cabin'] >= 700) & (train['Cabin'] < 800),'Cabin'] = 8

train.loc[(train['Cabin'] >= 800) & (train['Cabin'] < 900), 'Cabin' ] = 9

train.loc[(train['Cabin'] >= 900) & (train['Cabin'] < 1000), 'Cabin' ] = 10

train.loc[(train['Cabin'] >= 1000), 'Cabin' ] = 11



test.loc[test['Cabin'] < 100,'Cabin'] = 1

test.loc[(test['Cabin'] >= 100) & (test['Cabin'] < 200), 'Cabin' ] = 2

test.loc[(test['Cabin'] >= 200) & (test['Cabin'] < 300), 'Cabin' ] = 3

test.loc[(test['Cabin'] >= 300) & (test['Cabin'] < 400),'Cabin'] = 4

test.loc[(test['Cabin'] >= 400) & (test['Cabin'] < 500), 'Cabin' ] = 5

test.loc[(test['Cabin'] >= 500) & (test['Cabin'] < 600), 'Cabin' ] = 6

test.loc[(test['Cabin'] >= 600) & (test['Cabin'] < 700), 'Cabin' ] = 7

test.loc[(test['Cabin'] >= 700) & (test['Cabin'] < 800),'Cabin'] = 8

test.loc[(test['Cabin'] >= 800) & (test['Cabin'] < 900), 'Cabin' ] = 9

test.loc[(test['Cabin'] >= 900) & (test['Cabin'] < 1000), 'Cabin' ] = 10

test.loc[(test['Cabin'] >= 1000), 'Cabin' ] = 11
train.groupby(['Cabin','Survived']).count()
survived = train.loc[train['Survived'] == 1]

not_survived = train.loc[train['Survived'] == 0]
Pclass_survived = []

Pclass_not_survived = []



Sex_survived = []

Sex_not_survived = []



Age_survived = []

Age_not_survived = []



SibSp_survived = []

SibSp_not_survived = []



Name_survived = []

Name_not_survived = []



Ticket_survived = []

Ticket_not_survived = []



Fare_survived = []

Fare_not_survived = []



Cabin_survived = []

Cabin_not_survived = []



Embarked_survived = []

Embarked_not_survived = []



Parch_survived = []

Parch_not_survived = []



Title_survived = []

Title_not_survived = []



Fsize_survived = []

Fsize_not_survived = []





for index,row in train.iterrows():

    if row['Survived'] == 1:

        Pclass_survived.append(row['Pclass'])

        Age_survived.append(row['Age'])

        SibSp_survived.append(row['SibSp'])

        Name_survived.append(row['Name'])

        Ticket_survived.append(row['Ticket'])

        Fare_survived.append(row['Fare'])

        Cabin_survived.append(row['Cabin'])

        Embarked_survived.append(row['Embarked'])

        Parch_survived.append(row['Parch'])

        Sex_survived.append(row['Sex'])

        Title_survived.append(row['Title'])

        Fsize_survived.append(row['Fsize'])

    else:

        Pclass_not_survived.append(row['Pclass'])

        Age_not_survived.append(row['Age'])

        SibSp_not_survived.append(row['SibSp'])

        Name_not_survived.append(row['Name'])

        Ticket_not_survived.append(row['Ticket'])

        Fare_not_survived.append(row['Fare'])

        Cabin_not_survived.append(row['Cabin'])

        Embarked_not_survived.append(row['Embarked'])

        Parch_not_survived.append(row['Parch'])

        Sex_not_survived.append(row['Sex'])

        Title_not_survived.append(row['Title'])

        Fsize_not_survived.append(row['Fsize'])

        

        

        

        
train
train.corr()
score_table = []

for index,row in test.iterrows():

    score = 0

    if row['Pclass'] in Pclass_survived:

        score += Pclass_survived.count(row['Pclass']) / len(Pclass_survived) 

    if row['Age'] in Age_survived:

        score += Age_survived.count(row['Age']) / len(Age_survived) 

    if row['SibSp'] in SibSp_survived:

        score += SibSp_survived.count(row['SibSp']) / len(SibSp_survived) 

    if row['Name'] in Name_survived:

        score += Name_survived.count(row['Name']) / len(Name_survived) 

    if row['Ticket'] in Ticket_survived:

        score += Ticket_survived.count(row['Ticket']) / len(Ticket_survived) 

    if row['Fare'] in Fare_survived:

        score += Fare_survived.count(row['Fare']) / len(Fare_survived) 

    if row['Embarked'] in Embarked_survived:

        score += Embarked_survived.count(row['Embarked']) / len(Embarked_survived) 

    if row['Parch'] in Parch_survived:

        score += Parch_survived.count(row['Parch']) / len(Parch_survived) 

    if row['Sex'] in Sex_survived:

        score += Sex_survived.count(row['Sex']) / len(Sex_survived) 

    if row['Title'] in Title_survived:

        score += Title_survived.count(row['Title']) / len(Title_survived) 

    if row['Fsize'] in Fsize_survived:

        score += Fsize_survived.count(row['Fsize']) / len(Fsize_survived) 

    if row['Cabin'] in Cabin_survived:

        score += Cabin_survived.count(row['Cabin']) / len(Cabin_survived) 



    if row['Pclass'] in Pclass_not_survived:

        score -= Pclass_not_survived.count(row['Pclass']) / len(Pclass_not_survived) 

    if row['Age'] in Age_not_survived:

        score -= Age_not_survived.count(row['Age']) / len(Age_not_survived) 

    if row['SibSp'] in SibSp_not_survived:

        score -= SibSp_not_survived.count(row['SibSp']) / len(SibSp_not_survived) 

    if row['Name'] in Name_not_survived:

        score -= Name_not_survived.count(row['Name']) / len(Name_not_survived) 

    if row['Ticket'] in Ticket_not_survived:

        score -= Ticket_not_survived.count(row['Ticket']) / len(Ticket_not_survived) 

    if row['Fare'] in Fare_not_survived:

        score -= Fare_not_survived.count(row['Fare']) / len(Fare_not_survived) 

    if row['Embarked'] in Embarked_not_survived:

        score -= Embarked_not_survived.count(row['Embarked']) / len(Embarked_not_survived) 

    if row['Parch'] in Parch_not_survived:

        score -= Parch_not_survived.count(row['Parch']) / len(Parch_not_survived) 

    if row['Sex'] in Sex_survived:

        score -= Sex_not_survived.count(row['Sex']) / len(Sex_not_survived) 

    if row['Title'] in Title_not_survived:

        score -= Title_not_survived.count(row['Title']) / len(Title_not_survived) 

    if row['Fsize'] in Fsize_not_survived:

        score -= Fsize_not_survived.count(row['Fsize']) / len(Fsize_not_survived)

    if row['Cabin'] in Cabin_not_survived:

        score -= Cabin_not_survived.count(row['Cabin']) / len(Cabin_not_survived)



    score_table.append(score)

    if score > 0:

        test.loc[index,'Survived_predict'] = 1

    else:

        test.loc[index,'Survived_predict'] = 0



import matplotlib.pyplot as plt

plt.plot(sorted(score_table))

plt.ylabel('some numbers')

plt.show()
sum(test['Survived_predict'])
gender_submssion['Survived'] = test['Survived_predict']
gender_submssion['Survived'] = gender_submssion['Survived'].astype(int)
gender_submssion.head(60)
gender_submssion.to_csv('submission.csv',index = False)