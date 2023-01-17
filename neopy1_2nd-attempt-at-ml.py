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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
import re

def get_title(x):

    title = re.search(r'([A-Za-z]+)\.', x)

    if title:

        return title.group(1)

    else:

        return np.nan

    

train['Title'] = train['Name'].apply(get_title)

test['Title'] = test['Name'].apply(get_title)
train[np.isnan(train['Age'])]['Title'].value_counts()
test[np.isnan(test['Age'])]['Title'].value_counts()


#Fill in data for 'Master' title with MEAN

master_age_mean = train['Age'][(train['Title']=='Master')&(train['Age']>0)].mean()

train.loc[train[(train['Title']=='Master')&(train['Age'].isnull())].index, 'Age'] = master_age_mean

test.loc[test[(test['Title']=='Master')&(test['Age'].isnull())].index, 'Age'] = master_age_mean
#Fill in data for 'Mr' title with MEAN

master_age_mean = train['Age'][(train['Title']=='Mr')&(np.isfinite(train['Age']))].mean()

train.loc[train[(train['Title']=='Mr')&(train['Age'].isnull())].index, 'Age'] = master_age_mean

test.loc[test[(test['Title']=='Mr')&(test['Age'].isnull())].index, 'Age'] = master_age_mean
#Fill in data for 'Miss' title with MEAN

master_age_mean = train['Age'][(train['Title']=='Miss')&(np.isfinite(train['Age']))].mean()

train.loc[train[(train['Title']=='Miss')&(train['Age'].isnull())].index, 'Age'] = master_age_mean

test.loc[test[(test['Title']=='Miss')&(test['Age'].isnull())].index, 'Age'] = master_age_mean
#Fill in data for 'Mrs' title with MEAN

master_age_mean = train['Age'][(train['Title']=='Mrs')&(np.isfinite(train['Age']))].mean()

train.loc[train[(train['Title']=='Mrs')&(train['Age'].isnull())].index, 'Age'] = master_age_mean

test.loc[test[(test['Title']=='Mrs')&(test['Age'].isnull())].index, 'Age'] = master_age_mean
#Fill in data for 'Dr' title with MEAN

master_age_mean = train['Age'][(train['Title']=='Dr')&(np.isfinite(train['Age']))].mean()

train.loc[train[(train['Title']=='Dr')&(train['Age'].isnull())].index, 'Age'] = master_age_mean

test.loc[test[(test['Title']=='Dr')&(test['Age'].isnull())].index, 'Age'] = master_age_mean
#Fill in data for 'Ms' title with MEAN

master_age_mean = train['Age'][(train['Title']=='Ms')&(np.isfinite(train['Age']))].mean()

train.loc[train[(train['Title']=='Ms')&(train['Age'].isnull())].index, 'Age'] = master_age_mean

test.loc[test[(test['Title']=='Ms')&(test['Age'].isnull())].index, 'Age'] = master_age_mean
train = train.drop('Cabin', axis=1)

test = test.drop('Cabin', axis=1)
train[train['Ticket'] == '113781']
train['Embarked'] = train['Embarked'].fillna('S')
test.info()
train['Family'] = train['SibSp']+train['Parch']

test['Family'] = test['SibSp']+test['Parch']



train['Family'] = train['Family'].apply(lambda x: 1 if x>0 else 0)

test['Family'] = test['Family'].apply(lambda x: 1 if x>0 else 0)

train = train.drop(['SibSp', 'Parch', 'Name', 'PassengerId', 'Ticket'], axis=1)

test = test.drop(['SibSp', 'Parch', 'Name', 'PassengerId', 'Ticket'], axis=1)
map1 = {'male':0,

       'female':1}

train['Sex'] = train['Sex'].map(map1)

test['Sex'] = test['Sex'].map(map1)
train.head()
test.Title.value_counts()
Title_map = {'Mr':0,

            'Miss':1,

             'Ms':1,

            'Mrs':2,

            'Master':3,

            'Dr':0,

            'Rev':0,

            'Mlle':1,

            'Major':4,

            'Col':4,

            'Capt':4,

            'Jonkheer':5,

            'Don':5,

             'Dona':5,

            'Sir':5,

            'Mme':2,

            'Lady':5,

            'Countess':5}



train['Title'] = train['Title'].map(Title_map)

test['Title'] = test['Title'].map(Title_map)
test.head()
train = pd.get_dummies(train, columns=['Embarked'])

test = pd.get_dummies(test, columns=['Embarked'])
test[np.isnan(test['Fare'])]
f_mn = test[(test['Embarked_S'] == 1) & (test['Pclass'] == 3) & (test['Title'] == 0) & (test['Family'] == 0)]['Fare'].mean()

f_std = test[(test['Embarked_S'] == 1) & (test['Pclass'] == 3) & (test['Title'] == 0) & (test['Family'] == 0)]['Fare'].std()

test['Fare'] = test['Fare'].fillna(f_mn)
Y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
Y_pred
test = pd.read_csv('../input/test.csv')



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)