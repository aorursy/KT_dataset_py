

import warnings

warnings.filterwarnings('ignore')



import scipy as sp

import pandas as pd

import re



import matplotlib as mlt

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
PID = "PassengerId"

SURVIVED = "Survived"

PCLASS = "Pclass"

NAME = "Name"

SEX = "Sex"

AGE = "Age"

SIBSP = "SibSp"

PARCH = "Parch"

TICKET = "Ticket"

FARE = "Fare"

CABIN = "Cabin"

EMBARKED = "Embarked"

FAMILY_SIZE = "FamilySize"

IS_ALONE = "IsAlone"
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df[SEX] = train_df[SEX].map({'male':1,'female':0})

test_df[SEX] = test_df[SEX].map({'male':1,'female':0})

train_df[EMBARKED] = train_df[EMBARKED].map({'S':2, 'C':1, 'Q':0})

test_df[EMBARKED] = test_df[EMBARKED].map({'S':2, 'C':1, 'Q':0})



all_data = pd.concat([train_df, test_df])

mid_age = all_data[AGE].mean()

# mid_age = pd.concat[train_df[AGE], test_df[AGE]].mean()

mid_embarked= all_data[EMBARKED].mean() 

# mid_embarked = pd.concat[train_df[EMBARKED], test_df[EMBARKED]].mean()



def preprocess(data, is_train=True):

    x = data.drop(columns=PID)

    x.fillna(mid_age,inplace=True)

    x[AGE].fillna(mid_age,inplace=True)

    x[EMBARKED].fillna(mid_embarked,inplace=True)

    

    x[FAMILY_SIZE] = x[SIBSP] + x[PARCH] + 1

    x[IS_ALONE] = x[FAMILY_SIZE]==1

    x.drop(columns=[SIBSP, PARCH, CABIN, NAME, TICKET],inplace=True)

    

    



    if is_train:

        y = x['Survived']

        x = x.drop(columns='Survived')

    else:

        y = None

        

    return x,y



train,valid = train_test_split(train_df, train_size=0.7)



x,y = preprocess(train_df)

x
x,y = preprocess(train)

x_valid, y_valid = preprocess(valid)

x_test, _ = preprocess(test_df,False)
# clf = RandomForestClassifier()

clf = LinearSVC()
_ = clf.fit(x,y)
pred_valid = clf.predict(x_valid)

accuracy_score(y_valid.values, pred_valid)
pred = clf.predict(x_test)
sub = pd.DataFrame(test_df['PassengerId'])

sub['Survived'] = list(map(int, pred))

sub.to_csv('submission.csv', index=False)