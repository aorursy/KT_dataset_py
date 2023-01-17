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
from tensorflow import keras

from tensorflow.keras import layers

import tensorflow as tf

from keras import Sequential

from keras.layers import Dense

from keras import utils



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

import math



import xgboost as xgb
df = pd.read_csv('../input/house-price-prediction-challenge/train.csv')

df.head(5)

test = pd.read_csv('../input/house-price-prediction-challenge/train.csv', usecols=['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 

            'READY_TO_MOVE', 'RESALE'])



X = df[['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 

            'READY_TO_MOVE', 'RESALE']]
X.head(5)

X.dtypes



y = df.iloc[:,-1]

y.head(5)



test.head(5)
#################### BUILD XGBOOST MODEL ###################################3

#now build the xgboost learner

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)



xg_reg.fit(X, y)



preds = xg_reg.predict(test)

preds
#################### EVALUATE MODEL QUALITY ###############################

# Create the DMatrix: housing_dmatrix

housing_dmatrix = xgb.DMatrix(data=X, label=y)



# Create the parameter dictionary: params

params = {"objective":"reg:squarederror", "max_depth":4}



# Perform cross-validation: cv_results

cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)



# Print cv_results

print(cv_results)



# Extract and print final boosting round metric

print((cv_results["test-rmse-mean"]).tail(1))
###################### SAVE THE PREDICTIONS #########################

predictions = pd.DataFrame(preds)

predictions.rename(columns={0:'TARGET(PRICE_IN_LACS)'}, inplace=True)

predictions = predictions.astype('int32')



predictions.to_csv('my_submission.csv', index=False)

subm = pd.read_csv('my_submission.csv')

subm


