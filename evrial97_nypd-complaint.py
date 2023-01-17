# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/NYPD_Complaint_Data_Historic.csv')
df.head()
df.describe()
df.info
df.isna().sum()
df = df[pd.notnull(df['CMPLNT_FR_DT'])]
df['CMPLNT_FR_DT'].dtype
df['CMPLNT_FR_DT'] = pd.to_datetime(df['CMPLNT_FR_DT'],infer_datetime_format=True, errors='coerce')
df['CMPLNT_FR_DT'].describe()
max_year = pd.to_datetime('2015')

min_year = max_year - pd.DateOffset(years=3)
df2 = df[(df['CMPLNT_FR_DT'] >= min_year) & (df['CMPLNT_FR_DT'] <= max_year)]
df2.head()
df3 = df2.groupby('CMPLNT_FR_DT').agg({'CMPLNT_FR_DT': "count"})

df3.rename({'CMPLNT_FR_DT': 'N_CRIME'}, axis=1, inplace=True)

df3

# df2.groupby('CMPLNT_FR_DT')['CMPLNT_FR_DT'].count() ###The same
# Ускладнений варіант

df2.groupby(['CMPLNT_FR_DT', 'SUSP_RACE']).agg({'CMPLNT_FR_DT': "count"})
len(df3)
df3_train = df3[:-10]

df3_test = df3[-10:]
from statsmodels.tsa.ar_model import AR



# fit model

model = AR(df3_train)

model_fit = model.fit()

# make prediction

yhat = pd.Series(model_fit.predict(len(df3_train), len(df3_train) + 9))

print(yhat)
yhat
plt.plot(range(1, 11), df3_test, '.', label='true')

plt.plot(range(1,11), yhat, '.', label='predicted')

plt.legend(loc='best')

plt.show()
from statsmodels.tsa.arima_model import ARMA



model = ARMA(df3_train, order=(0, 1))

model_fit = model.fit(disp=False)

# make prediction

yhat = pd.Series(model_fit.predict(len(df3_train), len(df3_train) + 9))

print(yhat)
plt.plot(range(1, 11), df3_test, '.', label='true')

plt.plot(range(1,11), yhat, '.', label='predicted')

plt.legend(loc='best')

plt.show()
from statsmodels.tsa.arima_model import ARIMA



# fit model

model = ARIMA(df3_train, order=(1, 1, 1))

model_fit = model.fit(disp=False)

# make prediction

yhat = pd.Series(model_fit.predict(len(df3_train), len(df3_train) + 9))

print(yhat)
plt.plot(range(1, 11), df3_test, '.', label='true')

plt.plot(range(1,11), yhat, '.', label='predicted')

plt.legend(loc='best')

plt.show()