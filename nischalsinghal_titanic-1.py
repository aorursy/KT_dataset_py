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
import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline





print("modules for data visualization imported")
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

print("train and test dataset loaded")
print("features in train \n")

train.info()

print("features in test \n")

test.info()

train.head()
train.tail()
plt.subplot2grid((3,4), (0,3))

train.Sex[train.Survived == 1].value_counts(normalize=True).plot(kind="bar")
train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sb.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=10)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sb.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Pclass', bins=3)
grid = sb.FacetGrid(train, col='Survived', row='Pclass')

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sb.FacetGrid(train, row='Embarked')

grid.map(sb.pointplot, 'Pclass', 'Survived', 'Sex', alpha=.5, bins=20)

grid.add_legend();
#now puttiing code for learning model

train_dataset = train

train_dataset["Hypothesis"] = 0

train_dataset.loc[train_dataset.Sex == "female", "Hypothesis"] = 1



train_dataset["Result"] = 0

train_dataset.loc[train_dataset.Survived == train["Hypothesis"], "Result"] = 1



print(train_dataset["Result"].value_counts(normalize=True))
grid = sb.FacetGrid(train, row='Pclass', col='Sex')

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
def data_process(data):

    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())

    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    

    

    

    data = data.drop(['Fare'], axis=1)

    data = data.drop(['Ticket'], axis=1)

    data = data.drop(['Cabin'], axis=1)

    freq_port = train.Embarked.dropna().mode()[0]

    

    data['Embarked'] = data['Embarked'].fillna(freq_port)



    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    

    data = data.drop(['Name'], axis=1)

    

    data.loc[data["Sex"] == "male", "Sex"] = 0

    data.loc[data["Sex"] == "female", "Sex"] = 1

      

    data.loc[ data['Age'] <= 16, 'Age'] = int(0)

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[data['Age'] > 64, 'Age']

    

    return data

    

    

    

    
import utils



train = data_process(train)

train.head()
from sklearn import linear_model, preprocessing



target = train["Survived"].values

features = train[["Pclass", "Sex", "Age", "SibSp", "Parch"]].values



classfier = linear_model.LogisticRegression()

classifier_ = classfier.fit(features, target)



print(classifier_.score(features, target))
poly = preprocessing.PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(features)



classfier = linear_model.LogisticRegression()

classifier_ = classfier.fit(poly_features, target)



print(classifier_.score(poly_features, target))
from sklearn import tree



decision_tree = tree.DecisionTreeClassifier(random_state = 1)

decision_tree_ = decision_tree.fit(features, target)



print(decision_tree_.score(features, target))
from sklearn.ensemble import *

random_forest = RandomForestClassifier(n_estimators=100)



Y_train = train["Survived"]

X_train = train.drop("Survived", axis=1)

X_train = X_train.drop("PassengerId", axis=1)

X_train = X_train.drop("Hypothesis", axis=1)

X_train = X_train.drop("Result", axis=1)





X_train.head()

random_forest.fit(X_train, Y_train)
X_test = data_process(test)

X_test = X_test.drop("PassengerId", axis=1)

X_test.head()

predicted_value = random_forest.predict(X_test)
test_dataset_copy = pd.read_csv('../input/titanic/test.csv')

submission = pd.DataFrame({

        "PassengerId": test_dataset_copy["PassengerId"],

        "Survived": predicted_value

})



submission.to_csv('submission.csv', index=False)