# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train_data = train_data.fillna(train_data.mean())

train_data.Age = train_data.Age.round(1)

train_data.Fare = train_data.Fare.round(2)

train_data["Sex"] = pd.get_dummies(train_data.Sex)

train_data["Embarked"] = pd.get_dummies(train_data.Embarked)

train_data
test_data = test_data.fillna(test_data.mean())

test_data.Age = test_data.Age.round(1)

test_data.Fare = test_data.Fare.round(2)

test_data["Sex"] = pd.get_dummies(test_data.Sex)

test_data["Embarked"] = pd.get_dummies(test_data.Embarked)

test_data
train_data.corr(method='pearson')

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Sex","Age"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions_RF = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_RF})

output.to_csv('my_submission_forest.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.neighbors import KNeighborsClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Age", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X,y)

predictions_knn = knn.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_knn})

output.to_csv('my_submission_knn.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.tree import DecisionTreeClassifier



features = ["Pclass", "Sex", "Fare", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

TitanicTree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)

TitanicTree.fit(X,y)

predictions_DT = TitanicTree.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_DT})

output.to_csv('my_submission_tree.csv', index=False)

print("Your submission was successfully saved!")
from sklearn import metrics

import matplotlib.pyplot as plt



print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y, predictions_DT))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch","Age", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X,y)

yhat = LR.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': yhat})

output.to_csv('my_submission_log.csv', index=False)

print("Your submission was successfully saved!")

from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y, yhat)