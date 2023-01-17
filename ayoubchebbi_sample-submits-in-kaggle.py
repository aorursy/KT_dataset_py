# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = '../input/train.csv'
train_data = pd.read_csv(train) 

train_data.columns
train_data = train_data.fillna(method='bfill')

train_data.head()
train_features = ['Pclass', 'Fare', 'Sex','Parch']
X = train_data[train_features]
X['Sex'],_ =pd.factorize(X['Sex'])
X['Fare'] =np.int64(X['Fare'])

X.head()
target = train_data.Survived
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, target)
test_data = pd.read_csv('../input/test.csv') 
 # test_data = test_data.dropna(axis=0)
test_data=test_data.fillna(method='bfill')
X_test = test_data[train_features]
X_test.head()
X_test['Sex'],_ =pd.factorize(X_test['Sex'])
X_test['Fare'] =np.int64(X_test['Fare'])

X_test.head()
my_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': np.int64(melbourne_model.predict(X_test))})
# you could use any filename. We choose submission here
my_submission.to_csv('tree_submission.csv', encoding='utf-8', index=False)