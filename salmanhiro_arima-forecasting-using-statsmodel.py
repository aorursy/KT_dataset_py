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
df = pd.read_csv("/kaggle/input/exchange-rates/exchange_rates.csv")

df = df.drop(df.index[0:5]).dropna()

df.head()

df.columns
df.head()
df.dtypes
df[df.columns[0]]
df = df[df != 'ND']

df.dropna()
from datetime import datetime



df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]) 

df[df.columns[1:len(df.columns)]] = df[df.columns[1:len(df.columns)]].astype(float)

import matplotlib.pyplot as plt



print(len(df))

print(df.dtypes)
import seaborn as sns



for i in range(1,len(df.columns)):

    plt.figure(figsize=(15,4))

    sns.lineplot(x = df[df.columns[0]], y = df[df.columns[i]])
for i in range(1,len(df.columns)):

    plt.figure(figsize=(15,4))

    sns.lineplot(x = df[df.columns[0]], y = np.log(df[df.columns[i]]))
from statsmodels.tsa.stattools import adfuller



def adf_test(timeseries):

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

       dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
for i in range(1,len(df.columns)):

    df[df.columns[i]] = df[df.columns[i]].fillna(method='ffill')

    print('\n',df.columns[i])

    adf_test(df[df.columns[i]])
for i in range(1,len(df.columns)):

    plt.figure(figsize=(15,4))

    df[df.columns[i]] = df[df.columns[1]] - df[df.columns[i]].shift(1)

    df[df.columns[i]].dropna().plot()

    
plt.figure(figsize=(15,4))

df[df.columns[1]].dropna().plot()

adf_test(df[df.columns[1]].dropna())
from statsmodels.tsa.arima_model import ARIMA

# using 1,1,1 ARIMA Model

model = ARIMA(df[df.columns[1]].dropna(), order=(1,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())
plt.figure(figsize=(15,4))

residuals = pd.DataFrame(model_fit.resid)

residuals.plot(title="Residuals")

residuals.plot(kind='kde', title='Density')

plt.show()
data = df[df.columns[1]].dropna().values



size = int(len(data) * 0.7)

train, test = data[0:size], data[size:len(data)]

history = [x for x in train]

predictions = list()



for t in range(len(test)):

    model = ARIMA(history, order=(1,1,1))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

from sklearn.metrics import mean_squared_error



#error = mean_squared_error(test, predictions)

#print('Test MSE: %.3f' % error)

plt.figure(figsize=(15,4))

plt.plot(test, label = 'actual')

plt.plot(predictions, color='red', label = 'predicted')

plt.legend()

plt.show()