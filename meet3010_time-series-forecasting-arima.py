# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import the dataset
df = pd.read_csv('../input/daily-total-female-births-in-california-1959/daily-total-female-births-CA.csv')
df
df.dtypes
df['date'] = pd.to_datetime(df['date'])
df_birth
df.set_index('date', inplace=True)
df
df.plot()
df_birth_smooth = df.rolling(window=20).mean()
df_birth_smooth
# we will visualise the data
df.plot()
df_birth_smooth.plot()
# we will perform the shifting on the data.
df['First Level Shifting'] = df['births'].shift(1)
df
df.shape
# Mean Sqaured Error
from sklearn.metrics import mean_squared_error
df_birth = df.dropna()
df_birth.shape
df.shape
error = mean_squared_error(df_birth['births'],df_birth['First Level Shifting'])
error
np.sqrt(error)
# import library
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
df
df.drop('First Level Shifting', axis = 1, inplace=True)
# plot_acf is to identify the parameter Q.
# ARIMA(p,d,q)

plot_acf(df)
# plot_acf is to identify the parameter P.
# ARIMA(p,d,q)

plot_pacf(df)
# p = 2 or 3
# q = 3 or 4
# d = 1
df.shape
# we will now create a training and testing data for our model.
df_train = df[:330]
df_test = df[330:]
df_train.shape,df_test.shape
# import ARIMA
from statsmodels.tsa.arima_model import ARIMA
birth_model = ARIMA(df_train,order = (2,1,3))
birth_model_fit = birth_model.fit()
birth_model_fit.aic
# Forecasting 
birth_forecast = birth_model_fit.forecast(steps=35)[0]
birth_forecast
# Now we will again checked the mean square error 
# mean_squared_error?
error = mean_squared_error(df_test,birth_forecast)
error
np.sqrt(error)