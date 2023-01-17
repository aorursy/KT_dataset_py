import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

from sklearn import linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

"Delete the Column name Cabin because it has little information as of the values ar NaN"
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

train.loc[train['Sex']=='male', 'Sex'] = 1
train.loc[train['Sex']=='female', 'Sex'] = 0

test.loc[test['Sex']=='male', 'Sex'] = 1
test.loc[test['Sex']=='female', 'Sex'] = 0

train.loc[train['Embarked']=='C', 'Embarked'] = 0
train.loc[train['Embarked']=='Q', 'Embarked'] = 1
train.loc[train['Embarked']=='S', 'Embarked'] = 2

test.loc[test['Embarked']=='C', 'Embarked'] = 0
test.loc[test['Embarked']=='Q', 'Embarked'] = 1
test.loc[test['Embarked']=='S', 'Embarked'] = 2

train['Embarked'] = train['Embarked'].fillna(0)
test['Embarked'] = test['Embarked'].fillna(0)

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y_train = train['Survived']

X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
X,x,Y,y = train_test_split(X_train, Y_train, test_size=0.2)
# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X, Y)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X, Y)
predictions = clf.predict(x)
print(accuracy_score(y, predictions))
ids = test['PassengerId']
predictions = clf.predict(X_test)


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
output.head()
