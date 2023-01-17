# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm, skew



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dftrain = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv')

dftest = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv')

dftrain.drop('Id', axis = 1, inplace = True)

test_id = dftest.Id

dftest.drop('Id',axis = 1, inplace = True)
#NO NULL
dftrain['Open Date'] = dftrain['Open Date'].apply(lambda x: pd.to_datetime(x))

dftest['Open Date'] = dftest['Open Date'].apply(lambda x: pd.to_datetime(x))

def calc_days(dtime):

    now_time = pd.to_datetime('2015-01-01')

    result = now_time - dtime

    return int(result.total_seconds()/3600//24)

dftrain['days_since_open'] = dftrain['Open Date'].apply(lambda x: calc_days(x))

dftest['days_since_open'] = dftest['Open Date'].apply(lambda x: calc_days(x))
dftrain['revenue'] = dftrain['revenue'].astype(int)
#dftrain.groupby('City')['revenue'].agg(['count','mean'])
#P1 int testset has no 7,8,10,11,13,14,16 values skew 1.8

#P2 FLOAT skew 0.03 test set only few decima values between 1 and 2, 4 and 5, 

#P3 float skew 0.14 test set only some decimal values between 4 and 5

#P10 int64 skew 1.70 test set 80% values are 5, 15% values are 10, test set exactly like train set.

#P22 int64 skew 0.79 test set 25% values are 1, 20% 2, 15% 3 , 12% 4, 9% 5 gradually decreasing distribution

#P23 int64 skew 3.24 test set 40% values are 1, 15% 2, 12% 3, 9% 4, 14% 5, 3-4% each 10 15 20 25

#P10 train and test exactly thesame
#no log transform = P2, P3, P7,


for col in dftrain.columns:

    if (dftrain[col].dtype == int) | (dftrain[col].dtype == float):

        if col not in ['P2','P3', 'P7','revenue']:

            dftrain[col] = dftrain[col].map(lambda x:np.log1p(x))

            print (col)

            dftest[col] = dftest[col].map(lambda x: np.log1p(x))

            
#for col in dftrain.columns:

 #   if (dftrain[col].dtype == int) | (dftrain[col].dtype == float):

  #      print (col)

   #     print (skew(dftrain[col]))

    #    print (skew(np.log1p(dftrain[col])))
pd.get_dummies(dftrain).columns
pd.get_dummies(dftest).columns[50:]
#log transforming dftrain revenue

dftrain['revenue']= dftrain.revenue.apply(lambda x: np.log1p(x))
dftrain1 = pd.get_dummies(dftrain, drop_first = True)

dftest1 = pd.get_dummies(dftest,drop_first = True)
y = dftrain1.revenue

X = dftrain1.drop('revenue', axis = 1)

for col in dftest1.columns:

    if col not in X.columns:

        dftest1.drop(col, axis = 1, inplace = True)

for col in X.columns:

    if col not in dftest1.columns:

        X.drop(col, axis = 1, inplace = True)

        print (col)

cat_cols = ['City', 'Open Date', 'City']
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

X.drop('Open Date', axis = 1, inplace = True)

dftest1.drop('Open Date', axis = 1, inplace = True)



X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size = 0.8, test_size = 0.2)



model = XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.1)

model.fit(X_train, y_train, early_stopping_rounds = 10, eval_set=[(X_valid, y_valid)], verbose = 10)
final_pred = model.predict(dftest1)
sampledf = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')

sampledf.head()
sub = pd.DataFrame()
sub['Id'] = sampledf.Id

sub['Prediction'] = np.expm1(final_pred)
sub.to_csv('submission.csv', index = False)