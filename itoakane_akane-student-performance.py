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
train = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv')

test = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv')
train
train['schoolsup'] = train['schoolsup'].map({True : 1 , False : 2})

train['famsup'] = train['famsup'].map({True : 1 , False : 2})

train['paid'] = train['paid'].map({True : 1 , False : 2})

train['activities'] = train['activities'].map({True : 1 , False : 2})

train['nursery'] = train['nursery'].map({True : 1 , False : 2})

train['hogher'] = train['higher'].map({True : 1 , False : 2})

train['internet'] = train['internet'].map({True : 1 , False : 2})

train['romantic'] = train['romantic'].map({True : 1 , False : 2})
test['schoolsup'] = test['schoolsup'].map({True : 1 , False : 2})

test['famsup'] = test['famsup'].map({True : 1 , False : 2})

test['paid'] = test['paid'].map({True : 1 , False : 2})

test['activities'] = test['activities'].map({True : 1 , False : 2})

test['nursery'] = test['nursery'].map({True : 1 , False : 2})

test['hogher'] = test['higher'].map({True : 1 , False : 2})

test['internet'] = test['internet'].map({True : 1 , False : 2})

test['romantic'] = test['romantic'].map({True : 1 , False : 2})
train = pd.get_dummies(train, drop_first=True)

test = pd.get_dummies(test, drop_first=True)
X = train.drop('G3', axis = 1).values

y = train['G3'].values
import xgboost as xgb

model = xgb.XGBRegressor()

model.fit(X , y)
X = test.values

predict = model.predict(X)
submit = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv')

submit['G3'] = predict

submit.to_csv('@akane-student-performance-prediction-XGBoost.csv', index=False)
submit