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
train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()
train_data.shape
test_data.shape
train_data.dtypes
train_data.describe()
train_data.count()
train_data.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_data
# input median(Age) for na's in Age column

train_data['Age'] = train_data['Age'].fillna(value = train_data['Age'].median())
# input mode(Embarked) which is 'S' for na's in Embarked column

train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.count()
# input median(Age) for na's in Age column

test_data['Age'] = test_data['Age'].fillna(value = test_data['Age'].median())
# input median(Fare) for na's in Fare column

test_data['Fare'] = test_data['Fare'].fillna(value = test_data['Fare'].median())
test_data.count()
# feature selection

features = ['Pclass','Sex','SibSp','Parch','Age','Embarked']
train_X = train_data[features]

train_X
test_X = test_data[features]

test_X
train_y = train_data['Survived']

train_y
from sklearn.tree import DecisionTreeClassifier



decision_tree_model = DecisionTreeClassifier(random_state = 0)

X = pd.get_dummies(train_X)

decision_tree_model.fit(X, train_y)

tX = pd.get_dummies(test_X)

pred_y = decision_tree_model.predict(tX)

decision_tree_model_accuracy = round(decision_tree_model.score(X, train_y) * 100, 2)

decision_tree_model_accuracy
from sklearn.ensemble import RandomForestClassifier



random_forest_model = RandomForestClassifier(random_state = 0)

random_forest_model.fit(X,train_y)

predictions = random_forest_model.predict(tX)

random_forest_model_accuracy = round(random_forest_model.score(X, train_y) * 100, 2)

random_forest_model_accuracy
from sklearn import svm



clf = svm.SVC(kernel = 'poly',random_state = 0)

clf.fit(X, train_y)

pred_y = clf.predict(tX)

clf_accuracy = round(clf.score(X, train_y) * 100, 2)

clf_accuracy
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")