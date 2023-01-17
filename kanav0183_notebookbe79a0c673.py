# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv("../input/all_data.csv")

# Any results you write to the current directory are saved as output.

data.head()
data.dtypes


import datetime as dt

data['date'] = data.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']), axis = 1)

data['year'] = data.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']).year, axis = 1)

data['month'] = data.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']).month, axis = 1)

data['day'] = data.apply(lambda x: dt.datetime.fromtimestamp(x['timestamp']).day, axis = 1)

data.head(10)
data.describe()
def miss(x):

    return(sum(x.isnull()))

data.apply(miss)


import seaborn as sns

sns.heatmap(data.corr(), annot=True)

data['month'].value_counts()
train = data.loc[data['year'] != 2017 ]

test = data.loc[data['year'] == 2017 ]                   
import matplotlib.pyplot as plt

%matplotlib inline
plt.subplots(figsize=(15, 9))

plt.plot(data['date'],data['price_USD'])
train.groupby('month')['price_USD'].median().plot(kind = 'bar')
from sklearn.linear_model import LinearRegression
col = ['total_addresses', 'transactions']
train[col]
reg = LinearRegression()

reg.fit(train[col] , train['price_USD'])

k =reg.predict(test[col])

from sklearn.ensemble import RandomForestClassifier

#model= RandomForestClassifier(n_estimators=1000)

# Train the model using the training sets and check score

#model.fit(train[col] , train['price_USD'])

#Predict Ou

#r= model.predict(test[col])
from sklearn.metrics import mean_squared_error

mean_squared_error(test['price_USD'],k)
import xgboost as xgb



from sklearn import cross_validation, model_selection

xgbfolds = model_selection.KFold(n_splits=5)



xgb_dtrain = xgb.DMatrix(train[col] , train['price_USD'])

xgb_dtest = xgb.DMatrix(test['price_USD'])



xgb_params = {'learning_rate' : 0.03, 

             'subsample' : 0.7,

             'max_depth' : 5,

             'colsample_bytree' : 0.8,

              'objective': 'reg:linear',

              'eval_metric': 'rmse',

             'silent': 0

             }

xgb_obj = xgb.cv(params = xgb_params, dtrain = xgb_dtrain, early_stopping_rounds=10,

                       verbose_eval=True, show_stdv=False, folds = xgbfolds, num_boost_round = 9999)
xgb = xgb.train(params = xgb_params, dtrain = xgb_dtrain, num_boost_round = 80)



predictions = xgb.predict(xgb_dtest)



from sklearn.metrics import mean_squared_error

mean_squared_error(test['price_USD'],predictions)