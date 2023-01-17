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
import pandas as pd
import numpy as np
from IPython.display import display

dataset_train = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')
dataset_train = dataset_train.drop(axis=1,labels=['Cabin','PassengerId','Name','Ticket'])
dataset_test = dataset_test.drop(axis=1,labels=['Cabin','Name','Ticket'])
dataset_test_pid = dataset_test.loc[:,'PassengerId']
dataset_test = dataset_test.drop(labels='PassengerId',axis=1)

from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(dataset_train['Age'].values.reshape(-1,1))

dataset_train['Age'] = imputer.transform(dataset_train['Age'].values.reshape(-1,1))
dataset_test['Age'] =  imputer.transform(dataset_test['Age'].values.reshape(-1,1))

imputer = Imputer()
imputer = imputer.fit(dataset_test['Fare'].values.reshape(-1,1))
dataset_test['Fare'] = imputer.transform(dataset_test['Fare'].values.reshape(-1,1))

dataset_train = dataset_train.dropna()
dataset_train = pd.get_dummies(dataset_train)

dataset_train_x = dataset_train.drop(labels=['Survived'],axis=1)
dataset_train_y = dataset_train.loc[:,'Survived']
dataset_test = pd.get_dummies(dataset_test)

# GridSearchCV to find optimal n_estimators
from xgboost import XGBClassifier

# specify number of folds for k-fold CV
# n_folds = 5

# # parameters to build the model on
# parameters = {
#                 'subsample': [0.3, 0.6, 0.9],
#                 'n_estimators' : range(200,1600,200)
#                }

# instantiate the model
rf = XGBClassifier(max_depth=2,learning_rate=0.2,gamma = 1,
                   subsample = 0.9,n_jobs = -1,n_estimators = 5000,)

#fit tree on training data
# rf = GridSearchCV(rf, parameters, 
#                       cv=n_folds, 
#                      scoring="accuracy",verbose = 1,n_jobs=-1)
rf.fit(dataset_train_x, dataset_train_y)

predict = rf.predict(dataset_test)
display(rf.score(dataset_train_x,dataset_train_y))

output = pd.DataFrame({'PassengerId':dataset_test_pid,'Survived':predict})

output.to_csv("predictions.csv", encoding='utf-8',index=False)