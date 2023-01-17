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
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import seaborn as sns

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
df = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')
df.info()
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
sns.lineplot(legend = 'full' , data=df)
df1=df

df1['rolling_mean'] = df['#Passengers'].rolling(window = 12).mean()

df1['rolling_std'] = df['#Passengers'].rolling(window = 12).std()

sns.lineplot(data=df1, legend='full')
result = adfuller(df['#Passengers'])

print('ADF Statistic: {}'.format(result[0]))

print('p-value: {}'.format(result[1]))

print('Critical Values:')

for key, value in result[4].items():

    print('\t{}: {}'.format(key, value))
def rand_iterator():

    return np.random.randint(low = 1, high = 3, size = 4)



order_params=[]

results_store=[]



for i in range(50):

    order_params.append(rand_iterator())

    

print(order_params)



#param_grid={'order':order_params}

#model = ARIMA()

#model_fit=GridSearchCV(model,param_grid,n_jobs=-1,cv=3,scoring='neg_mean_absolute_error')

#model_fit.fit(df)



for order_val in order_params:

    model=SARIMAX(df['#Passengers'],order=order_val)

    results= model.fit(disp=0)

    results_store.append(results)

    print('\tORDER = {}: AIC = {}'.format(order_val, results.aic))

    

    

plt.plot(results.fittedvalues, color='red')

plt.plot(df['#Passengers'])
from sklearn.metrics import mean_absolute_error

size = int(len(df) * 0.66)

train, test = df['#Passengers'][0:size], df['#Passengers'][size:len(df)]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

    model = SARIMAX(history, order=(1,1,2,1))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_absolute_error(test, predictions)

print('Test MAPE: %.3f ' % error)

# plot

#plt.plot(test)

#plt.plot(predictions, color='red')
#sns.lineplot(data=test, legend='full')

#sns.lineplot(data=predictions, legend='full', color='red')
size = int(len(df) * 0.66)

df2=df[size:len(df)]

df2.head()
df2['test']=test

df2['predictions']=predictions
df2.tail()
plt.plot(df2['test'])

plt.plot(df2['predictions'], color='red')
forecast=results.forecast(steps=5)
data=pd.concat([df2,forecast])
data=data.drop(['rolling_mean','rolling_std','test'],axis=1)
data[30:].plot()