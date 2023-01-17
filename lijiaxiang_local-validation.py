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
os.getcwd()
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")

sample_submission  = pd.read_csv("../input/sample_submission.csv")
train.columns.tolist()
train.describe()
sample_submission.head()
from sklearn.model_selection import train_test_split

validation_train, validation_test = train_test_split(train, test_size=0.3, random_state=123)
import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt



naive_prediction = np.mean(validation_train.sales)



# Assign naive prediction to all the holdout observations

validation_test['pred'] = naive_prediction



# Measure the local RMSE

rmse = sqrt(mean_squared_error(validation_test['sales'], validation_test['pred']))

print('Validation RMSE for Baseline I model: {:.3f}'.format(rmse))
test['sales'] = naive_prediction

print(test['sales'].head())

test[['id','sales']].to_csv("baseline_v1.0.0.csv", index=False)
grouping_prediction = validation_train.groupby(['store']).sales.mean()



# Assign naive prediction to all the holdout observations

validation_test['pred'] = validation_test.store.map(grouping_prediction)



# Measure the local RMSE

rmse = sqrt(mean_squared_error(validation_test['sales'], validation_test['pred']))

print('Validation RMSE for Baseline II model: {:.3f}'.format(rmse))
test['sales'] = test.store.map(grouping_prediction)

print(test['sales'].sample(n=5))

test[['id','sales']].to_csv("baseline_v1.1.0.csv", index=False)
rf = RandomForestRegressor(n_estimators=10, random_state=123)



# Train a model

rf.fit(X=validation_train[['store', 'item']], y=validation_train['sales'])



# Get predictions for the test set

validation_test['pred'] = rf.predict(validation_test[['store', 'item']])



# Measure the local RMSE

rmse = sqrt(mean_squared_error(validation_test['sales'], validation_test['pred']))

print('Validation RMSE for Baseline III model: {:.3f}'.format(rmse))
test['sales'] = rf.predict(test[['store', 'item']])

print(test['sales'].sample(n=5))

test[['id','sales']].to_csv("baseline_v1.2.0.csv", index=False)