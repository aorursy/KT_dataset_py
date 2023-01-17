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
import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import pandas as pd

import itertools
data=pd.read_excel('../input/Superstore.xls')

data.head()
"""

There are several categories in the Superstore sales data, 

we start from time series analysis and forecasting for furniture sales.

"""

furniture = data.loc[data['Category'] == 'Furniture']

furniture.head()
cols_to_be_dropped = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 

        'Customer ID', 'Customer Name', 'Segment', 'Country', 

        'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 

        'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols_to_be_dropped, axis=1, inplace=True)

furniture.head()
furniture = furniture.sort_values('Order Date')

furniture.isnull().sum()

# we need to group the sales by date

furniture_modified=furniture.groupby('Order Date')['Sales'].sum().reset_index()

furniture_modified.head()
furniture_modified=furniture_modified.set_index('Order Date')

y = furniture_modified['Sales'].resample('MS').mean() #reducing Sales to the average sales per month

y.head()
y.plot(figsize=(15, 6))

plt.show()
# below code is to decompose the data series

import statsmodels.api as sm

from pylab import rcParams

plt.style.use('fivethirtyeight')

rcParams['figure.figsize'] = 15, 10

decomposition = sm.tsa.seasonal_decompose(y, model='additive')

fig = decomposition.plot()

plt.show()
# Define the p, d and q parameters to take any value between 0 and 2

p = d = q = range(0, 2)



# Generate all different combinations of p, q and q triplets

pdq = list(itertools.product(p, d, q))



# Generate all different combinations of seasonal p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
import warnings

warnings.filterwarnings("ignore") # specify to ignore warning messages

results={}

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)



            result = mod.fit()

            if result.aic not in results:

                results[result.aic]=[param,param_seasonal]

            #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, result.aic))

        except:

            continue
min_aic=min(results)

print(min_aic)

results[min_aic]
dataset=y.reset_index()



train_dataset=dataset.iloc[:36,:]

test_dataset=dataset.iloc[36:,:]
import numpy as np

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_error

from statistics import mean



tscv = TimeSeriesSplit()

print(tscv)

list_MSE=[] # A list to store means square error for each fold of validation



# using TimeSeriesSplit for kfold validation

for train_index, validation_index in tscv.split(train_dataset):

#     print("TRAIN:", train_index, "TEST:", test_index)

    y_train, y_validate = train_dataset.iloc[train_index], train_dataset.iloc[validation_index]

    mod = sm.tsa.statespace.SARIMAX(y_train.set_index("Order Date"),

                                order=(0, 1, 1),

                                seasonal_order=(0, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

    model=mod.fit()

    y_validate.set_index("Order Date")

    y_pred=model.forecast(6, alpha=0.05) # 6 samples to be forecast with 95% conf



    mse = mean_squared_error(y_validate.values[:,1], y_pred.values) # y_test is the true values against which "predicted values" are evaluated.

    list_MSE.append(mse)

    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))



print("The average of MSE is {}".format(mean(list_MSE)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mean(list_MSE)), 2)))
# Now it is time to build the final model of SARIMA 



Final_Model=sm.tsa.statespace.SARIMAX(train_dataset.set_index("Order Date"),

                                order=(0, 1, 1),

                                seasonal_order=(0, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

Final_Model=Final_Model.fit()

test_dataset.set_index("Order Date")

predictions=Final_Model.forecast(12, alpha=0.05) # 12 samples of year 2017 to be forecast with 95% conf



mse = mean_squared_error(test_dataset.values[:,1], predictions.values) # y_test is the true values against which "predicted values" are evaluated.

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
print(Final_Model.summary().tables[1])

#testing the model

# pred = fitted_model.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

pred_ci = Final_Model.conf_int()

pred_ci



ax = y.plot(label='observed')

predictions.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)



# ax.fill_between(pred_ci.index,

#                 pred_ci.iloc[:, 0],

#                 pred_ci.iloc[:, 1], color='k', alpha=.2)



ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()



plt.show()