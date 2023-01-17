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
# Collect sales data from each category into one dataframe

categoriesDF = CalendarDF[['Food','Hobbies','Household']]

categoriesDF.corr(method = 'pearson')

categoriesDF.corr(method = 'spearman')

categoriesDF.corr(method = 'kendall')
from statsmodels.tsa.seasonal import seasonal_decompose



Food.index = CalendarDF['date']



# Split food sales data into train and test 

foodTrain = Food['20110129':'20160410']

foodTest = Food['20160411':'20160522']



# Drop 0 sales values to prepare data for multiplicative seasonal decomposition

foodTrain = foodTrain[foodTrain[foodTrain.columns[0]] !=0]



# Seasonal decomposition

result = seasonal_decompose(foodTrain, model = 'multiplicative', extrapolate_trend = 'freq', freq = 7) # frequency set to weekly



# Store seasonality component of decomposition

seasonal = result.seasonal.to_frame()

seasonal_index = result.seasonal[-7:].to_frame()



# Merge the train data and the seasonality 

foodTrain = foodTrain.merge(seasonal, how = 'left', on = foodTrain.index , left_index = True, right_index = True)
# Building the SARIMAX model

# I use the Pyramid Arima package to perform an auto-SARIMAX forecast



!pip install pmdarima

import pmdarima as pm



#SARIMAX Model setting the exogenous variable to weekly seasonality 

sxmodel = pm.auto_arima(foodTrain[foodTrain.columns[0]], exogenous= foodTrain[['seasonal']],

                           start_p=1, start_q=1,

                           test='adf',

                           max_p=3, max_q=3, m=7,

                           start_P=0, seasonal=True,

                           d=None, D=1, trace=True,

                           error_action='ignore',  

                           suppress_warnings=True, 

                           stepwise=True)



sxmodel.summary()
# Forecasting using the SARIMAX model

import matplotlib.pyplot as plt



n_periods = 42

fitted, confint = sxmodel.predict(n_periods = n_periods,  exogenous= np.tile(seasonal_index['seasonal'], 6).reshape(-1,1),  return_conf_int = True)



index_of_fc = pd.date_range(foodTest.index[0], periods = n_periods, freq = 'D')



# make series for plotting purpose

fitted_series = pd.Series(fitted, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)

upper_series = pd.Series(confint[:, 1], index=index_of_fc)



# Plot

plt.plot(foodTest)

plt.plot(fitted_series, color='darkgreen')

plt.fill_between(lower_series.index, 

                 lower_series, 

                 upper_series, 

                 color='k', alpha=.15)



plt.title("SARIMA - Total Sales of TX_1")

plt.show()
# data engineering for event_name_1

CalendarDF['isweekday'] = [1 if wday >= 3 else 0 for wday in CalendarDF.wday.values]

CalendarDF['isweekend'] = [0 if wday > 2 else 1 for wday in CalendarDF.wday.values]

CalendarDF['holiday_weekend'] = [1 if (we == 1 and h not in [np.nan]) else 0 for we,h in CalendarDF[['isweekend','event_name_1']].values]

CalendarDF['holiday_weekday'] = [1 if (wd == 1 and h not in [np.nan]) else 0 for wd,h in CalendarDF[['isweekday','event_name_1']].values]



# one-hot-encoding event_name_1

CalendarDF = pd.get_dummies(CalendarDF, columns=['event_name_1'], prefix=['holiday'], dummy_na=True)



Food = CalendarDF['Food']

Food.index = CalendarDF['date']



# Section out the columns created by encoding and concat with Food dataframe

temp = CalendarDF.iloc[:,16:50]

temp.index = CalendarDF['date']

Food = pd.concat([Food, temp], axis = 1)



foodTrain = Food['20110129':'20160410']

foodTest = Food['20160411':'20160522']
# Build the SARIMAX model

sxmodel_event = pm.auto_arima(foodTrain[foodTrain.columns[0]], exogenous= foodTrain.iloc[:,1:],

                           start_p=1, start_q=1,

                           test='adf',

                           max_p=3, max_q=3, m=7,

                           start_P=0, seasonal=True,

                           d=None, D=1, trace=True,

                           error_action='ignore',  

                           suppress_warnings=True, 

                           stepwise=True)



sxmodel_event.summary()
# Forecast

n_periods = 42

event_predict, confint = sxmodel_event.predict(n_periods = n_periods,  exogenous= foodTest.iloc[:,1:],  return_conf_int = True)



index_of_fc = pd.date_range(foodTest.index[0], periods = n_periods, freq = 'D')



# make series for plotting purpose

fitted_series = pd.Series(event_predict, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)

upper_series = pd.Series(confint[:, 1], index=index_of_fc)



# Plot

#plt.plot(foodTrain)

plt.plot(foodTest)

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
symmetric_mean_absolute_percentage_error(foodTest[foodTest.columns[0]], fitted) #sMAPE of SARIMAX with forced seasonality
symmetric_mean_absolute_percentage_error(foodTest[foodTest.columns[0]], event_predict) #sMAPE of SARIMAX with event_name_1