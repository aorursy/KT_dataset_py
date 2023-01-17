# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/air-passengers/AirPassengers.csv')
df.head()
df.shape
df.isna().sum()
df.info()
from datetime import datetime
df['Month']=pd.to_datetime(df['Month'],infer_datetime_format=True)

df.info()
index = df.set_index('Month',inplace=False)
index.head()
index.plot()
index.rename(columns={'#Passengers':'Passengers'},inplace=True)
simple_ma = index['Passengers'].rolling(window=12).mean()

simple_std = index['Passengers'].rolling(window=12).std()
plt.title('Data vs Mean vs Std')

plt.plot(index,color='yellow')

plt.plot(simple_ma, color='blue')

plt.plot(simple_std,color='red')
from statsmodels.tsa.stattools import adfuller
df_test = adfuller(index['Passengers'],autolag='AIC')

df_test
from statsmodels.tsa.stattools import kpss
kpss_test = kpss(index['Passengers'],'ct')

kpss_test
index_log = np.log(index)

ma_log = index_log.rolling(window=11).mean()

std_log = index_log.rolling(window=12).std()
plt.title("Logarithmic Values vs MA vs Std")

plt.plot(index_log)

plt.plot(ma_log)

plt.plot(std_log)
index_new = index_log - ma_log

index_new.dropna(inplace=True)
ma_new = index_new.rolling(window=11).mean()

std_new = index_new.rolling(window=12).std()
plt.title("New Index Values vs MA vs Std")

plt.plot(index_new)

plt.plot(ma_new)

plt.plot(std_new)
df_test_new = adfuller(index_new['Passengers'],autolag='AIC')

df_test_new
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(index_log , order = (2,1,2))

results_ARIMA = model.fit(disp=-1)
predictions = pd.Series(results_ARIMA.fittedvalues, copy=True)

predictions.head()
