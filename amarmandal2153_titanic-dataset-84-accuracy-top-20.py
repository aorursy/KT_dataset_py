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


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
pd.set_option('display.max_columns', None)
print(train.shape, test.shape)

train.head()
data = pd.concat([train, test], axis = 0)
data
data.drop(['PassengerId','Name'], axis = 1, inplace = True)
data.Ticket.unique()
data.drop(['Ticket'], axis = 1, inplace = True)
data['Survived'].replace(np.nan, 0, inplace = True)
data.Survived = data.Survived.astype('int')
data.head()
for feature in data.columns:

    print(f'{feature} = {data[feature].sort_values().unique()}')
data.isnull().sum()

data.drop(['Cabin'], axis = 1, inplace = True)
data.head()
# categorical
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

data['Sex'] = label.fit_transform(data['Sex'])
data.head()
pd.get_dummies(data['Embarked'], drop_first = True)
data[['Q','S']] = pd.get_dummies(data['Embarked'], drop_first = True)
data.head()
data.drop(['Embarked'], axis = 1, inplace = True)
data.isnull().sum()
# Numerical
data.Age = data.Age.fillna(data["Age"].median())
data.isnull().sum()
data.Fare = data.Fare.fillna(data["Fare"].median())
data.isnull().sum()
data['Age'].unique()
# Separation
X_train = data.iloc[:891, :]

X_test = data.iloc[891:, :]
X_testt = X_test.drop(['Survived'], axis = 1)

X_testt = X_testt.values
y_train = X_train[['Survived']]

X_train = X_train.drop(['Survived'], axis = 1)
X = X_train.values



y = y_train.values

X
# Modelling
# Splitting data into Train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train
# XGBoost

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Confusion matrix



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
# Applying K-Fold_Cross_Validation in Kernel_svm



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)

print(accuracies.mean(), accuracies.std())
accuracies.mean()
# Hypertunning
from sklearn.model_selection import GridSearchCV



parameter = {'learning_rate' : [0.05, 0.075, 0.1, 0.2, 0.3, 0.4]}

grid = GridSearchCV(estimator= model, param_grid= parameter, verbose = 10)

grid.fit(X_train, y_train)


print(grid.best_score_)
# gamma





parameter = {'gamma' : [0, 0.05, 0.1]}

model = XGBClassifier(learning_rate = 0.05)

grid = GridSearchCV(estimator= model, param_grid= parameter, verbose = 10)

grid.fit(X_train, y_train)
grid.best_params_
grid.best_score_
# max_depth



parameter = {'max_depth' : [2, 3, 4, 5, 6]}

model = XGBClassifier(learning_rate = 0.05, gamma = 0)

grid = GridSearchCV(estimator= model, param_grid= parameter, verbose = 10)

grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_
# n_estimators



parameter = {'n_estimators' : [50,75,100,125,150]}

model = XGBClassifier(learning_rate = 0.05, gamma = 0, max_depth = 4)

grid = GridSearchCV(estimator= model, param_grid= parameter, verbose = 10)

grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_
# min_child_weight



parameter = {'min_child_weight' : [1, 2, 3, 4, 5]}

model = XGBClassifier(learning_rate = 0.05, gamma = 0, max_depth = 4, n_estimators = 100, random_state = 42)

grid = GridSearchCV(estimator= model, param_grid= parameter, verbose = 10)

grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_
# final model



model = XGBClassifier(learning_rate = 0.05, gamma = 0, max_depth = 4, n_estimators = 100,min_child_weight= 3, random_state = 42)

grid = GridSearchCV(estimator= model, param_grid= parameter, verbose = 10)

grid.fit(X_train, y_train)
grid.best_score_