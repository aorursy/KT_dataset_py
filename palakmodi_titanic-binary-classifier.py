# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_df.columns
train_df.dtypes
train_df = train_df.drop('Name',axis=1)
train_df = train_df.drop('Ticket',axis=1)
train_df = train_df.drop('Cabin',axis=1)
test_df = test_df.drop('Name',axis=1)
test_df = test_df.drop('Ticket',axis=1)
test_df = test_df.drop('Cabin',axis=1)
X = pd.get_dummies(train_df)
test_X = pd.get_dummies(test_df)
final_train, final_test = X.align(test_X, join='left', axis=1)
X.head()
y = X['Survived']
X = X.drop('Survived',axis=1)
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
X = my_imputer.fit_transform(X)
test_X = my_imputer.transform(test_X)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X,y)
predictions = model.predict(test_X)
my_submission = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)