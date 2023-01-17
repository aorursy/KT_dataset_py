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
df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

df

age_mean = df.Age.mean()

age_mean
melbourne_data = df.dropna(axis=0)

from sklearn.tree import DecisionTreeRegressor



y = melbourne_data['Survived']



feature_columns = ['Survived', 'Age', 'SibSp', 'Parch','Fare']

X = melbourne_data[feature_columns]



# Define model. Specify a number for random_state to ensure same results each run

melbourne_model = DecisionTreeRegressor(random_state=1)



# Fit model

melbourne_model.fit(X, y)



print("First in-sample predictions:", melbourne_model.predict(X.head()))

print("Actual target values for those homes:", y.head().tolist())



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex4 import *

print("Setup Complete")

from sklearn.model_selection import train_test_split



# fill in and uncomment

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

iowa_model = DecisionTreeRegressor(random_state = 1)



# Fit iowa_model with the training data.

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

print (val_predictions[:5])

# print the top few actual prices from validation data

print (val_y[:5])
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y,val_predictions)



# uncomment following line to see the validation_mae

print(val_mae)
val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
tirange = list(range(100, 1001, 100))

print(tirange)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBRegressor, XGBClassifier



results = {}

for n in tirange:

    print('Range=', n)

    # model = RandomForestClassifier(n_estimators=n)

    model = XGBClassifier(n_estimators=n, learning_rate=0.05, n_jobs=4)

    model.fit(train_X, train_y, early_stopping_rounds=20,

              eval_set=[(val_X, val_y)], 

             verbose=False)

    preds = model.predict(val_X)

    accu = accuracy_score(y_true=val_y, y_pred=preds)

    f1 = f1_score(y_true=val_y, y_pred=preds, average='micro')

    print(classification_report(y_true=val_y, y_pred=preds))



    results[n] = f1