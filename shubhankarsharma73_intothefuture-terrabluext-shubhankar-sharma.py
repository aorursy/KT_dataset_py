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
import matplotlib.pyplot as plt

import seaborn as sns

import fbprophet as Prophet

from statsmodels.tsa.statespace.varmax import VARMAX
train_df = pd.read_csv('../input/into-the-future/train.csv')

test_df =pd.read_csv('../input/into-the-future/test.csv')

train_df_copy = train_df.copy()

test_df.copy = test_df.copy
train_df.head()
train_df.tail()
test_df.head()
train_df.describe()
test_df.describe()
train_df.info()
test_df.info()
train_df['time'] = pd.to_datetime(train_df.time)

test_df['time'] = pd.to_datetime(test_df.time)
train_df.set_index('time', inplace=True)
train_df['feature_1'].plot(figsize=(20,10))
train_df['feature_2'].plot(figsize=(20,10))
print(train_df.columns)

print(test_df.columns)
print(train_df.head())

print(test_df.head())
test_df.set_index('time', inplace=True)
print(train_df.head())

print(test_df.head())
test_df.set_index('id', inplace=True)

train_df.set_index('id', inplace=True)
print(train_df.head())

print(test_df.head())
#We can now split the training dataset in train and valid

#Due to such a small dataset we'll keep the validition set at 10% of the training set

train = train_df[:int(0.9*len(train_df))]

valid = train_df[int(0.9*len(train_df)):]
print(train.shape)

print(valid.shape)
train.head()
from statsmodels.tsa.statespace.varmax import VARMAX



model1 = VARMAX(train)

model1_fit = model1.fit()
preds = model1_fit.forecast(steps=len(valid))

preds.head(10)
#importing MAE and MSE for metrics

from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import mean_absolute_error as MAE
import math

rmse=math.sqrt(MSE(preds,valid))

print('Mean absolute error is: '+ str(MAE(preds,valid)))

print('Root Mean Squared error is: ' + str(rmse))
full_model = VARMAX(train_df)

full_model_fit = full_model.fit()
preds = full_model_fit.forecast(steps=len(test_df))
predictions = pd.DataFrame(preds)

predictions.to_csv('results.csv')
#Work_In_Progress

#Check_Pt_1: RMSE = 220.48511984812345 | MAE = 145.34370904420948

#With VARMAX

#Would be using other models soon