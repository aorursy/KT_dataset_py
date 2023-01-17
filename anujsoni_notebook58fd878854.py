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
airline = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv',index_col='Month',parse_dates=True)
airline.head()
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

result = seasonal_decompose(airline['#Passengers'],model='multiplicative')
result.plot()
from pmdarima import auto_arima

import warnings

warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(airline['#Passengers'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',   # we don't want to know if an order does not work 

                          suppress_warnings = True,  # we don't want convergence warnings 

                          stepwise = True)           # set to stepwise 

  

# To print the summary 

stepwise_fit.summary()
train = airline.iloc[:len(airline)-12] 

test = airline.iloc[len(airline)-12:] # set one year(12 months) for testing 

  

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set 

from statsmodels.tsa.statespace.sarimax import SARIMAX 

  

model = SARIMAX(train['#Passengers'],  

                order = (0, 1, 1),  

                seasonal_order =(2, 1, 1, 12)) 

  

result = model.fit() 

result.summary()
start = len(train) 

end = len(train) + len(test) - 1

  

# Predictions for one-year against the test set 

predictions = result.predict(start, end, 

                             typ = 'levels').rename("Predictions") 

  

# plot predictions and actual values 

predictions.plot(legend = True) 

test['#Passengers'].plot(legend = True)
from sklearn.metrics import mean_squared_error 

from statsmodels.tools.eval_measures import rmse 

  

# Calculate root mean squared error 

print(rmse(test["#Passengers"], predictions)) 

  

# Calculate mean squared error 

print(mean_squared_error(test["#Passengers"], predictions))
# Train the model on the full dataset 

model = SARIMAX(airline['#Passengers'],  

                        order = (0, 1, 1),  

                        seasonal_order =(2, 1, 1, 12)) 

result = model.fit() 

  

# Forecast for the next 3 years 

forecast = result.predict(start = len(airline),  

                          end = (len(airline)-1) + 3 * 12,  

                          typ = 'levels').rename('Forecast') 

  

# Plot the forecast values 

airline['#Passengers'].plot(figsize = (12, 5), legend = True) 

forecast.plot(legend = True) 
forecast.to_csv('submission.csv')