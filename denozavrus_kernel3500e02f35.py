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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline
train_path = "../input/titanic/train.csv"

test_path = "../input/titanic/test.csv"

train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)

train_data.head()
test_data.head()
def RandomForestScore (X_train, Y_train, X_test, Y_test,estimators):

    model = RandomForestClassifier(n_estimators = estimators, max_depth = 10, random_state = 0)

    model.fit(X_train, Y_train)

    prediction = model.predict(X_test)

    acc = accuracy_score (Y_test, prediction)

    print (acc)

    return acc
def CrossValRFScore(X, Y, estimators):

    model = RandomForestClassifier (n_estimators = estimators, max_depth = 10, random_state = 0)

    my_imputer = SimpleImputer(strategy = 'median')

    pipeline = Pipeline (steps = [

        ('preprocessor', my_imputer), 

        ('model',model)

    ])

    score = cross_val_score(pipeline, X, Y, cv = 5, scoring = 'accuracy')

    print (score.mean())

    return score.mean()
train_data.isnull().any()
features = ['Pclass', 'Sex','Age','SibSp','Parch','Embarked']

X = train_data[features]

Y = train_data.Survived

X.head()
X['Family'] = X['SibSp'] + X['Parch']

X = X.drop(['SibSp','Parch'],axis = 1)

X.head()
X.Sex = X.Sex.map({'male': 0, 'female': 1}).astype(int)

X.Embarked = X.Embarked.fillna('S')

X.Embarked = X.Embarked.map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

X.head()
X.head()
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
my_imputer = SimpleImputer(strategy = 'mean')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns
imputed_X_train.isnull().any()

imputed_X_valid.isnull().any()
result = {}

for est in range (10, 300, 20): 

    result[est] = RandomForestScore(X_train = imputed_X_train, 

                                    Y_train = y_train, 

                                    X_test = imputed_X_valid, 

                                    Y_test = y_valid, 

                                    estimators = est)

plt.plot(result.keys(), result.values())

plt.show()

#(X_train, Y_train, X_test, Y_test,estimators)

#55 - best no arguments for depth

#115 - best max depth = 10 acc = 0.8324

Y.head()
result = {}

for est in range (10, 300, 20): 

    result [est] = CrossValRFScore (X, Y, est)

plt.plot(result.keys(), result.values())

plt.show()

#(X, Y, estimators)
features = ['Pclass', 'Sex','Age','SibSp','Parch','Embarked']

X_test = test_data[features]

X_test.head()

X_test['Family'] = X_test['SibSp'] + X_test['Parch']

X_test = X_test.drop(['SibSp','Parch'],axis = 1)

X_test.head()
X_test.Sex = X_test.Sex.map({'male': 0, 'female': 1}).astype(int)

X_test.Embarked = X_test.Embarked.fillna('S')

X_test.Embarked = X_test.Embarked.map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

X_test.head()
my_imputer = SimpleImputer(strategy = 'median')

X_imputed = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

imputed_X_test.columns = X_test.columns

X_imputed.columns = X.columns
imputed_X_test.head()
imputed_X_test.isnull().any()
X_imputed.head()
Y.head()
X_imputed.isnull().any()
test_data.head()
model = RandomForestClassifier (n_estimators = 110, max_depth = 10, random_state = 0)

model.fit(X_imputed,Y)

preds = model.predict(imputed_X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':preds})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")