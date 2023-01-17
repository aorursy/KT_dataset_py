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
df = pd.read_csv('/kaggle/input/time-series-datasets/Electric_Production.csv')

df.head()
df.dtypes
df['DATE'] = pd.to_datetime(df['DATE'])

df.head()
df.index= df['DATE']

df = df.drop(columns=['DATE'],axis=1)

df.head()
from statsmodels.tsa.stattools import adfuller

adf = adfuller(df.IPG2211A2N)

print('p-value:',adf[1])
from matplotlib import pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(10,5))

ax = plt.subplot(1,2,2)

plt.plot(df.IPG2211A2N)

plot_acf(df.IPG2211A2N);
plt.plot(df.IPG2211A2N.diff().dropna())

plot_acf(df.IPG2211A2N.diff().dropna());
plt.plot(df.IPG2211A2N.diff().diff())

plot_acf(df.IPG2211A2N.diff().diff());
from statsmodels.graphics.tsaplots import plot_pacf

plt.plot(df.IPG2211A2N)

plot_pacf(df.IPG2211A2N);
plt.plot(df.IPG2211A2N.diff().dropna())

plot_pacf(df.IPG2211A2N.diff().dropna());
plt.plot(df.IPG2211A2N.diff().dropna())

plot_acf(df.IPG2211A2N.diff().dropna());
from statsmodels.tsa.arima_model import ARIMA

arima = ARIMA(df.IPG2211A2N,order=(1,1,1))

model = arima.fit()

print(model.summary())
plt.figure(figsize=(10,10))

model.plot_predict();
train_set = df[0:365]

test_set = df[365:]
arima = ARIMA(train_set,order=(1,1,1))

model = arima.fit()

print(model.summary())
fcast,se,confidencebands = model.forecast(32,alpha=0.01)
pred_set = pd.DataFrame(data=fcast,columns=['Value']) 

pred_set.index = test_set.index

pred_set.tail()
plt.figure(figsize=(20,10))

plt.plot(train_set,label='Training examples')

plt.plot(test_set,label='Original Values')

plt.plot(pred_set,label='Predicted forecast')

plt.legend()

plt.xlabel('Year')

plt.ylabel('Electricity production')

model.plot_predict(1,500);