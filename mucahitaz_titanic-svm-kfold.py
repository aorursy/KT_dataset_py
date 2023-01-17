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

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
test2 = pd.read_csv("../input/titanic/test.csv")
import warnings

warnings.simplefilter('ignore')



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import cross_val_score, KFold,StratifiedKFold,train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import roc_auc_score
print('Train columns with null values:\n', train.isnull().sum())

print("-" * 10)

print('Test columns with null values:\n', test.isnull().sum())

print("-" * 10)
dropping = ['PassengerId', 'Ticket','Cabin','Fare']

train.drop(dropping,axis=1, inplace=True)

test.drop(dropping,axis=1, inplace=True)

train.head()
data = [train,test]

for dataset in data:

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)

    Embarked = np.zeros(len(dataset))

    Embarked[dataset['Embarked']== 'C'] = 1

    Embarked[dataset['Embarked']== 'Q'] = 2

    Embarked[dataset['Embarked']== 'S'] = 3

    dataset['Embarked'] = Embarked
train['Age'].fillna(train['Age'].median(), inplace = True)

test['Age'].fillna(test['Age'].median(), inplace = True)
print('Train columns with null values:\n', train.isnull().sum())

print("-" * 10)

print('Test columns with null values:\n', test.isnull().sum())

print("-" * 10)
data = [train,test]

for dataset in data:

    dataset['Age'] = dataset['Age']

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



train['Age'].value_counts()
data = [train,test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)
data = [train,test]

for dataset in data:

    sex = np.zeros(len(dataset))

    sex[dataset['Sex']== 'male'] = 1

    sex[dataset['Sex']== 'female'] = 0

    dataset['Sex'] = sex
data = [train,test]

for dataset in data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1    
data = [train,test]

for dataset in data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
train_y=train['Survived']

train_ft=train.drop('Survived',axis=1)

kf = StratifiedKFold(n_splits=10)

print(train_ft.head())

print(train_y.head())
from sklearn.svm import SVC, LinearSVC

svc = SVC(C = 45, gamma = 0.03)

svc.fit(train_ft, train_y) 



acc_SVM = cross_val_score(svc,train_ft,train_y,cv=kf)

print(acc_SVM.mean())
predictions = svc.predict(test)

print(predictions)
submission = pd.DataFrame({ 'PassengerId': test2['PassengerId'],

                            'Survived': predictions })

submission.to_csv('submission.csv', index = False)