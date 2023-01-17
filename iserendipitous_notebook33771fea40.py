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
import numpy as np

import pandas as pd

import seaborn as sns
X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

print(X_train.shape)

print(X_test.shape)

print(X_train.dtypes)
X_train['Sex'] = X_train[X_train['Sex'].map({'female':0, 'male':1})]

X_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

if len(X_train.Embarked[X_train.Embarked.isnull()])>0:

    X_train.Embarked[X_train.Embarked.isnull()]=X_train.Embarked.dropna().mode().values

Ports=list(enumerate(np.unique(X_train['Embarked'])))

Ports_dict = {name:i for i, name in Ports}

X_train.Embarked = X_train.Embarked.map(lambda x: Ports_dict[x]).astype(int)

median_age = X_train['Age'].dropna().median()

if len(X_train.Age[X_train.Age.isnull()])>0:

    X_train.loc[(X_train.Age.isnull()), 'Age']=median_age

print(X_train.dtypes, X_train.shape)
print(X_train[pd.isnull(X_train).any(axis=1)])
# we try to cut the age group to get a better glimpse, we will add a new column to training data here

bins=[0,10,18,30,50,100]

group_names = ['kid','teenager', 'youth', 'middle_aged', 'Senior']

X_train['ages'] = pd.cut(X_train.Age, bins,labels=group_names )

print(X_train.dtypes, X_train.head(5))
sns.swarmplot(x='Survived', y='Age', hue='ages', data=X_train)

sns.factorplot(x='ages', y='Survived', col='Pclass', data=X_train, kind='bar')
y_train = np.ravel(X_train.loc[:,['Survived']])

X_train.drop(['Survived', 'ages'], axis=1, inplace=True)

print(X_train.shape)
# the same cleaning procedure for test data

if len(X_test.Embarked[X_test.Embarked.isnull()])>0:

    X_test.Embarked[X_test.Embarked.isnull()] = X_test.Embarked.dropna().mode().values

ports = list(enumerate(np.unique(X_test['Embarked'])))

ports_dict = {name:i for i, name in ports}

X_test.Embarked = X_test.Embarked.map(lambda x: ports_dict[x]).astype(int)

median_age_test = X_test['Age'].dropna().median()

if len(X_test.Age[X_test.Age.isnull()])>0:

    X_test.loc[(X_test.Age.isnull()), 'Age']=median_age_test



median_fare = X_test['Fare'].dropna().median()

if len(X_test.Fare[X_test.Fare.isnull()])>0:

    X_test.loc[(X_test.Fare.isnull()), 'Fare']=median_fare

X_test['Sex']=X_test['Sex'].map({'female':0, 'male':1})

ids = X_test['PassengerId']

X_test.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

print(X_test.shape, X_test.dtypes)
#we will add Parch and Sibling into a single column in both training and testing data

X_train['Family'] = X_train['Parch'] + X_train['SibSp']

X_train.drop(['Parch', 'SibSp'], axis=1, inplace=True)

X_test['Family'] = X_test['Parch'] + X_test['SibSp']

X_test.drop(['Parch', 'SibSp'], axis=1, inplace=True)

print(X_train.shape, X_test.shape)
train_data = X_train.values

test_data = X_test.values

from sklearn.svm import SVC

svc = SVC(C=1, kernel='rbf').fit(train_data, y_train)

y_predict=svc.predict(test_data)

print(svc.score(train_data, y_train))
import csv

predictions_file = open('myfirstsubmission.csv', 'wb')

data = {'PassengerId':ids, 'Survived': y_predict}

frame = pd.DataFrame(data)

frame.to_csv('myfirstsubmission.csv', index=False)