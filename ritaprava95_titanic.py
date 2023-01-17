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
train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)

index_col = test.index

train_X = train.drop(['Survived','Name', 'Ticket', 'Cabin'], axis=1)
train_X = pd.get_dummies(train_X).values

train_y = train.loc[:,'Survived'].values

test = test.drop(['Ticket', 'Name', 'Cabin'],axis=1)
test = pd.get_dummies(test).values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_X = imp.fit_transform(train_X)
test = imp.fit_transform(test)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
model = SVC()
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(model, parameters)
searcher.fit(train_X,train_y)
print("Best CV params", searcher.best_params_)
pred = searcher.predict(test)

predicted_df = pd.DataFrame(columns=['Survived'],
                            index = index_col,
                            data=pred)
predicted_df.to_csv('predicted_1.csv')