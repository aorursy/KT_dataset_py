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
!
#del data`

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/train.csv')



train_test = [train, test]



print(train_test[0].shape)

print(train_test[1].shape)



import re



for data in train_test:

    data['Initial'] = ''

    init = []

    p = re.compile('([A-Za-z]+\.)')

    for value in data['Name']:

        init.append(p.search(value).group())

    data['Initial'] = init



print(data['Initial'])
pre = ['Dr.',  'Dona.','Rev.', 'Col.', 'Major.', 'Mile.', 'Sir.', 'Jonkheer.', 'Don.', 'Mme.', 'Countess.','Capt.', 'Mlle.', 'Lady.', 'Master.']

aft = ['Other.','Other.','Other.','Other.','Other.','Other.','Other.','Other.','Other.','Other.','Other.', 'Other.','Other.', 'Other.','Other.' ]

for data in train_test:

    data['Initial'].replace(pre, aft, inplace=True)
for data in train_test:

    data.loc[(data.Age.isnull()) &(data.Initial=='Miss.'), 'Age'] = 21

    data.loc[(data.Age.isnull()) &(data.Initial=='Mr.'), 'Age'] = 32

    data.loc[(data.Age.isnull()) &(data.Initial=='Mrs.'), 'Age'] = 35

    data.loc[(data.Age.isnull()) &(data.Initial=='Ms.'), 'Age'] = 28

    data.loc[(data.Age.isnull()) &(data.Initial=='Other.'), 'Age'] = 20
mapping = {

    'Mr.' : 1,

    'Miss.' : 2,

    'Mrs.': 3,

    'Ms.': 4,

    'Other.': 5

}



for data in train_test:

    data['Initial'] = data['Initial'].map(mapping).astype(int)

    

mapping = {

    'female' : 1,

    'male' : 0

}



for data in train_test:

    data['Sex'] = data['Sex'].map(mapping).astype(int)

    

for data in train_test:

    data['Embarked'].fillna('S', inplace=True)



mapping = {

    'S' : 0,

    'C' : 1,

    'Q' : 2

}

for data in train_test:

    data['Embarked'] = data['Embarked'].map(mapping).astype(int)



for data in train_test:

    data.loc[data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[data['Age'] > 64, 'Age'] = 4



train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)



for data in train_test:

    data['Fare'].fillna(train['Fare'].median(), inplace=True)

    

train['FareBand'] = pd.qcut(train['Fare'], 4)



for data in train_test:

    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31.0), 'Fare'] = 2

    data.loc[data['Fare'] > 31.0, 'Fare'] = 3



train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)



for data in train_test:

    data['Family'] = data['SibSp'] + data['Parch']



for data in train_test:

    data.loc[data['Family'] == 0, 'Family'] = 0

    data.loc[(data['Family'] >= 1) & (data['Family'] < 4), 'Family'] = 1

    data.loc[(data['Family'] >= 4) & (data['Family'] < 7), 'Family'] = 2

    data.loc[data['Family'] >= 7, 'Family'] = 3



drop_list = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']

train = train.drop(drop_list, axis=1)

test = test.drop(drop_list, axis=1)



train = train.drop(['PassengerId', 'FareBand'], axis=1)



X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop(["PassengerId", 'Survived'], axis=1).copy()

y_test = test['Survived']
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_pred_log_reg = clf.predict(X_test)

print("acc ", clf.score(X_train, y_train))
!kaggle competitions submit -c titanic -f submission.csv -m "Message"
ls -al