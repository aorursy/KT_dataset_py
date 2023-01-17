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

import pandas as pd 



CalendarDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", header=0)

SalesDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv", header=0) #June 1st Dataset
import os, psutil



pid = os.getpid()

py = psutil.Process(pid)

memory_use = py.memory_info()[0] / 2. ** 30

print ('memory GB:' + str(np.round(memory_use, 2)))
CalendarDF['date'] = pd.to_datetime(CalendarDF.date)



TX_1_Sales = SalesDF[['TX_1' in x for x in SalesDF['store_id'].values]]

TX_1_Sales = TX_1_Sales.reset_index(drop = True)

TX_1_Sales.info()
# Generate MultiIndex for easier aggregration.

TX_1_Indexed = pd.DataFrame(TX_1_Sales.groupby(by = ['cat_id','dept_id','item_id']).sum())

TX_1_Indexed.info()
# Aggregate total sales per day for each sales category

Food = pd.DataFrame(TX_1_Indexed.xs('FOODS').sum(axis = 0))

Hobbies = pd.DataFrame(TX_1_Indexed.xs('HOBBIES').sum(axis = 0))

Household = pd.DataFrame(TX_1_Indexed.xs('HOUSEHOLD').sum(axis = 0))

Food.info()
# Merge the aggregated sales data to the calendar dataframe based on date

CalendarDF = CalendarDF.merge(Food, how = 'left', left_on = 'd', right_on = Food.index)

CalendarDF = CalendarDF.rename(columns = {0:'Food'})

CalendarDF = CalendarDF.merge(Hobbies, how = 'left', left_on = 'd', right_on = Hobbies.index)

CalendarDF = CalendarDF.rename(columns = {0:'Hobbies'})

CalendarDF = CalendarDF.merge(Household, how = 'left', left_on = 'd', right_on = Household.index)

CalendarDF = CalendarDF.rename(columns = {0:'Household'})

CalendarDF.head(10)
# Drop dates with null sales data

CalendarDF = CalendarDF.drop(CalendarDF.index[1941:])

CalendarDF.reset_index(drop = True)
# Aggregate all sales to perform analysis 

totalSales = pd.DataFrame(columns = ['Sales'])

totalSales['Sales'] = CalendarDF['Food'] + CalendarDF['Hobbies'] + CalendarDF['Household']

totalSales.index = CalendarDF['date']

totalSales.head(10)
import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import pyplot

from statsmodels.tsa.seasonal import seasonal_decompose



# Decomposing the sales data from the full time period

fullDecompose = seasonal_decompose(totalSales, model = 'additive')

fullDecompose.plot() # plotting the decomposition

pyplot.show()
# Decomposing the sales data from 2011

is_2011 = CalendarDF.year == 2011

Calendar2011 = CalendarDF[is_2011] # pull 2011 data from CalendarDF



# Collect total sales data from 2011

Sales2011 = Calendar2011['Food'] + Calendar2011['Hobbies'] + Calendar2011['Household'] 

Sales2011.reset_index(drop = True)

Sales2011.index = Calendar2011['date']



# Decomposition

Decompose2011 = seasonal_decompose(Sales2011, model = 'additive')

Decompose2011.plot()

pyplot.show()
#Decomposing the sales data from February 2011

is_February = Calendar2011.month == 2

February2011 = Calendar2011[is_February] # pull February data from Calendar2011



# Collect total sales data from February 2011

februarySales = February2011['Food'] + February2011['Hobbies'] + February2011['Household']

februarySales.reset_index(drop = True)

februarySales.index = February2011['date']



# Decomposition

februaryDecompose = seasonal_decompose(februarySales, model = 'additive')

februaryDecompose.plot()

pyplot.show()
# To build an ARIMA Model, I have to find the 'p', 'q', and 'd' terms. Details found on Selva Prabhakaran's guide



# Begin by checking if the Total Sales time series is stationary using the Augmented Dickey Fuller test



from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



statTest = adfuller(totalSales)

print('ADF Statistic: %f' % statTest[0])

print('p-value: %f' % statTest[1])
# Since the Total Sales time series does not seem to be stationary (p-value > .05), I perform differencing to find the 'd' and 'q' terms of the ARIMA model



statTest = adfuller(totalSales.diff().dropna())

print('p-value: %f' % statTest[1]) # p-value < .05, set 'd' = 1



# Plot the first differencing to find the 'q' term

plot_acf(totalSales.diff().dropna()) # Since lag 3 is well above the significance region, set 'q' = 3
# To find the P term, find the first significant lag by plotting the Partial Autocorrelation of the differenced time series

plot_pacf(totalSales.diff().dropna()) # the first lag is significant, p = 1
# Building the ARIMA model

from statsmodels.tsa.arima_model import ARIMA



# Segment the total sales data into Train and Test datasets

salesTrain = totalSales['20110130':'20160410']

salesTest = totalSales['20160411':'20160522']



# Create the model (p,d,q) using p = 1, d =1, q = 3

model = ARIMA(salesTrain.values, order = (1,1,3)) 

model_fit = model.fit(disp = 0)

print(model_fit.summary())
model = ARIMA(salesTrain.values, order = (1,1,2)) 

model_fit = model.fit(disp = 0)

print(model_fit.summary())
# Actual vs Fitted

model_fit.plot_predict(dynamic = False)

plt.show()
# Plotting the residuals

residuals = pd.DataFrame(model_fit.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
# Using the model to forecast

fitted = model.fit(disp=-1)

fc, se, conf = fitted.forecast(42, alpha = .05) # predicting the total sales of the next 42 days



# Make each forecast value as a pandas Series to plot confidence interval

fc_series = pd.Series(fc, index = salesTest.index)

lower_series = pd.Series(conf[:,0], index = salesTest.index)

upper_series = pd.Series(conf[:,1], index = salesTest.index)



# To improve visualization, I only plot the salesTest data against the forecasted values

plt.plot(salesTest, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
# I use the Pyramid Arima package to perform an auto-SARIMA forecast

!pip install pmdarima

import pmdarima as pm



# We set the seasonality ('m') to 7 in order to account for the weekly seasonality present in the data

smodel = pm.auto_arima(salesTrain, start_p = 1, start_q = 1, test = 'adf',

                      max_p = 3, max_q = 3, m = 7,

                      start_P = 0, seasonal = True, d = None, D = 1,

                      trace = True, error_action = 'ignore', suppress_warnings = True,

                      stepwise = True)



smodel.summary()
# Forecasting using the SARIMA model

n_periods = 42 # forecasting the next 42 days of sales

fitted, confint = smodel.predict(n_periods = n_periods, return_conf_int = True)

index_of_fc = pd.date_range(salesTest.index[0], periods = n_periods, freq = 'D') # setting freq = 'D' for daily indexing



# Making series for plotting purposes

fitted_series = pd.Series(fitted, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)

upper_series = pd.Series(confint[:, 1], index=index_of_fc)
# Plotting the salesTest data against the forecasted values

plt.plot(salesTest)

plt.plot(fitted_series, color='darkgreen')

plt.fill_between(lower_series.index, 

                 lower_series, 

                 upper_series, 

                 color='k', alpha=.15)



plt.title("SARIMA - Total Sales of TX_1")

plt.show()
#Accuracy metrics

def symmetric_mean_absolute_percentage_error(actual,forecast):

    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual)/(np.abs(actual)+np.abs(forecast)))



def mean_absolute_error(actual, forecast):

    return np.mean(np.abs(actual - forecast))



def naive_forecasting(actual, seasonality):

    return actual[:-seasonality]



def mean_absolute_scaled_error(actual, forecast, seasonality):

    return mean_absolute_error(actual, forecast) / mean_absolute_error(actual[seasonality:], naive_forecasting(actual, seasonality))
# Using SMAPE to evaluate the ARIMA model

symmetric_mean_absolute_percentage_error(salesTest['Sales'], fc)
# Using MASE to evaluate the SARIMA model

mean_absolute_scaled_error(salesTest.values, fitted, seasonality = 7) # set seasonality to 7 to account for weekly seasonality