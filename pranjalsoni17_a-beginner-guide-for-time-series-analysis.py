import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
data = pd.read_csv('../input/gold-price-data/gld_price_data.csv')

gold_data = data[['Date','GLD']]

gold_data.columns = ['date','gold_price']

gold_data.head()
gold_data.index = pd.to_datetime(gold_data['date'])

gold_data.drop('date',axis=1,inplace = True)
gold_data['2018-01']
import matplotlib.pyplot as plt

plt.plot(gold_data)

plt.xlabel('Year')

plt.ylabel('Price [USD]')

plt.title('Gold Prices')

plt.grid()

plt.show()
print(gold_data['gold_price'].autocorr())
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(gold_data['gold_price'],lags=20,alpha=0.5)

plt.show()
from statsmodels.tsa.stattools import adfuller

results = adfuller(gold_data['gold_price'])

print('The p-value of the test on prices is: ' + str(results[1]))

from statsmodels.tsa.stattools import adfuller

results = adfuller(gold_data.pct_change().dropna())

print('The p-value of the test on prices is: ' + str(results[1]))

#here we take diff() to make gold data stationary.

diff_gold_data = gold_data.diff().dropna()

diff_gold_data.plot()
from statsmodels.tsa.arima_model import ARMA

model = ARMA(gold_data,order=(1,0)) #here order attribute is used to select AR model

result = model.fit()

print(result.summary())
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(gold_data['gold_price'],lags=20,alpha=0.10)

plt.show()
parameters = [i for i in range(1,6)]

bic_values = []

for p in parameters:

    model = ARMA(gold_data,order=(p,0))

    result = model.fit()

    bic_values.append(result.bic)

plt.plot(bic_values)

plt.xticks(parameters)

plt.xlabel('No. of Parameter')

plt.ylabel('BIC value')

plt.show()
model = ARMA(gold_data,order=(1,0))

result = model.fit()

'''Start the forecast 10 data points before the end of the 2290 point series at 2280,and 

end the forecast 10 data points after the end of the series at point 2300'''

result.plot_predict(start=2280, end=2300)

plt.xticks()

plt.plot()



#order 1 MA model

from statsmodels.tsa.arima_model import ARMA

model = ARMA(gold_data,order=(0,1)) #here order attribute is used to select MR model

result = model.fit()

print(result.summary())
#partial autocorrelation function plot

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(gold_data, lags=20)

plt.show()
model = ARMA(gold_data,order=(0,1))

result = model.fit()

'''Start the forecast 10 data points before the end of the 2290 point series at 2280,and 

end the forecast 10 data points after the end of the series at point 2300'''

result.plot_predict(start=2280, end=2300)

plt.xticks()

plt.plot()

model = ARMA(gold_data,order=(1,1)) # order =(1,1) is used to select order 1 ARMA model

result = model.fit()

'''Start the forecast 10 data points before the end of the 2290 point series at 2280,and 

end the forecast 10 data points after the end of the series at point 2300'''

result.plot_predict(start=2280, end=2300)

plt.xticks()

plt.plot()