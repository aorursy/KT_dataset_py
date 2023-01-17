# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns

%matplotlib inline









# import all models



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
train_df = train_data.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_data.drop(['Ticket', 'Cabin'], axis=1)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name','PassengerId'], axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_df['Age']=train_df['Age'].fillna(train_df['Age'].mean())

test_df['Age']=test_df['Age'].fillna(train_df['Age'].mean())
combine = [train_df, test_df]
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


X_train = pd.get_dummies(train_df[features])

Y_train = train_df["Survived"]

X_test  = pd.get_dummies(test_df[features])

X_train.shape, Y_train.shape, X_test.shape
X_test[X_test.isna().any(axis=1)]
X_test['Fare']=X_test['Fare'].fillna(X_test['Fare'].mean())
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]





decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

predictions = decision_tree.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")