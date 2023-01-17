# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

import itertools

import numpy as np 

import matplotlib.pyplot as plt 

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd 

import statsmodels.api as sm 

import matplotlib 



matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
df = pd.read_excel('../input/Superstore.xls')
# Forecaset furniture sales

furniture = df.loc[df['Category'] == 'Furniture']
# Check category timestamp 

furniture['Order Date'].min(), furniture['Order Date'].max()



df.head()
# Pre-proc 1

    # remove cols

    # aggregate sales by date 

    # check completeness 

    

    

# Grab cols 

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 

        'Country', 'City', 'State', 'Postal Code', 'Region',

       'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 

       'Profit']





# Drop 

furniture.drop(cols, axis = 1, inplace = True) 



# Sort

furniture = furniture.sort_values('Order Date')



furniture.isnull().sum()
# Aggregate

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()



furniture.head()
# Index with time series data 



furniture = furniture.set_index('Order Date')

furniture.index
# Use start of each month as timestamp 

# Use avg daily sales / month



# Set preferred calender frequency to Month Start in resample

    # https://stackoverflow.com/questions/17001389/pandas-resample-documentation

    

y = furniture['Sales'].resample('MS').mean()



y['2017']
# Plot y



y.plot(figsize = (15, 6))



plt.show()
# Sales low at start, high at end of years 



# Seasonality pattern
# Time series decomposition 

    # trend / seasonality / noise 

    

from pylab import rcParams 



rcParams['figure.figsize'] = 18, 8 



decomp = sm.tsa.seasonal_decompose(y, model = 'additive')



fig = decomp.plot()



plt.show()
# Sales is unstable with no stable pattern apart from seasonality 
# ARIMA forecasting 

    # Autoregressive Integrated Move Average 

    

# Param selection for ARIMA time series model 

    # seasonal model 

    

# pdq account for seasonality, trend, noise 



# Example 

p = d = q = range(0, 2)



pdq = list(itertools.product(p, d, q))



seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



print('Examples of param combinations for seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1],

                              seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1],

                              seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2],

                              seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2],

                              seasonal_pdq[4]))
# Use grid search to find optimal set of params for best model performance

    # Use lowest AIC for specified data

    

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y, # data

                                          order = param, 

                                          seasonal_order = param_seasonal,

                                          enforce_stationarity = False,

                                          enforce_invertibility = False)

            

            results = mod.fit()

            

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
# 297.78 = lowest AIC value 

    # = optimal option 

        # SARIMAX(1, 1, 1)x(1, 1, 0, 12) = AIC 297 



# Fit ARIMA model

mod = sm.tsa.statespace.SARIMAX(y, # data

                              order = (1, 1, 1),

                              seasonal_order = (1, 1, 0, 12),

                              enforce_stationarity = False,

                              enforce_invertibility = False)



results = mod.fit()



print(results.summary().tables[1])
# Run model diagnostics 

    # check for unusual behaviour

    

results.plot_diagnostics(figsize=(16, 8))



plt.show()
# Diagnostics suggest model redisuals are almost normally distributed
# Validate forecasts 

    # understand accuracy of forecast

        # compare predicted sales to real sales of time series

        

# Start prediction at 2017-01-01 to end 

    # Preserve pre-2017 



pred = results.get_prediction(start = pd.to_datetime('2017-01-01'),

                             dynamic = False)



pred_ci = pred.conf_int()



ax = y['2014':].plot(label = 'observed')

pred.predicted_mean.plot(ax = ax, 

                        label = "forecast",

                        alpha = .7,

                        figsize = (14, 7))



ax.fill_between(pred_ci.index,

               pred_ci.iloc[:, 0],

               pred_ci.iloc[:, 1],

               color = 'k',

               alpha = .2)



ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()



plt.show()
# forecasts align with true values very well

    # showing seasonality and upward trend towards end of year 
# MSE 



y_forecast = pred.predicted_mean

y_actual = y['2017-01-01':]



mse = ((y_forecast - y_actual) ** 2).mean()



print('MSE of forecast is {}'.format(round(mse, 2)))



    # lower MSE = better fit 

    

    # MSE = measure of estimator quality 

        # measures avg squared difference between estimated and actual 

        

    # Always non negative
# RMSE 



print('RMSE of forecast is {}'.format(round(np.sqrt(mse), 2)))

    

    # Indicate how concentrated data is around line of best fit 

    

    # Standard deviation of residuals (prediction errors)

    

    # model is able to forecast avg daily sales in the test set within 151.64 of real sales

    

    # daily sales range = 400 to > 1200 
# Plotting forecasts 



pred_1 = results.get_forecast(steps = 100)

pred_ci = pred_1.conf_int()



ax = y.plot(label = 'observed', figsize = (14, 7))



pred_1.predicted_mean.plot(ax = ax, label = 'Forecast')



ax.fill_between(pred_ci.index,

               pred_ci.iloc[:, 0],

               pred_ci.iloc[:, 1], 

                color = 'k', alpha = .25)



ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')



plt.legend()

plt.show()

# Shows seasonality 

# Confidence in values diminishes over time 

# Confidence intervals (grey) expands as forecasts moves further into the future
# Pre-proc 2 



furniture = df.loc[df['Category'] == 'Furniture']

office = df.loc[df['Category'] == 'Office Supplies']
furniture.shape, office.shape
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 

        'Country', 'City', 'State', 'Postal Code', 'Region',

       'Product ID', 'Category', 'Sub-Category', 'Product Name', 

        'Quantity', 'Discount', 

        'Profit']



furniture.drop(cols, axis = 1, inplace = True)

office.drop(cols, axis = 1, inplace = True)



furniture = furniture.sort_values('Order Date')

offifce = office.sort_values('Order Date')



furniture = furniture.groupby("Order Date")["Sales"].sum().reset_index()

office = office.groupby("Order Date")["Sales"].sum().reset_index()
furniture.head()
office.head()
# Combine dfs 

# Compare sales by day 

    # INNER JOIN on Order Date 



# Index with time series 



furniture = furniture.set_index('Order Date')

office = office.set_index('Order Date')



# Preferred freq = Month start

    # AVG sales / month

    

y_furniture = furniture['Sales'].resample('MS').mean()

y_office = office['Sales'].resample('MS').mean()



# DF preparation



furniture = pd.DataFrame({'Order Date': y_furniture.index,

                         'Sales': y_furniture.values})



office = pd.DataFrame({'Order Date': y_office.index, 

                       'Sales': y_office.values})



# Merge 



store = furniture.merge(office, how = 'inner', on = 'Order Date')



store.rename(columns = {'Sales_x': 'furniture_sales', 

                       'Sales_y': 'office_sales'}, inplace = True)



store.head()
# Plot time series on 1 plot 



plt.figure(figsize=(20, 8))

plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label = 'furniture')

plt.plot(store['Order Date'], store['office_sales'], 'r-', label = 'office_supplies')

plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Sales of 2')

plt.legend();
# Similar seasonsal pattern



# AVG daily sales Furn > office for most months
# First time office passed furn



first_time = store.ix[np.min(list(np.where(store['office_sales'] > store['furniture_sales'])[0])),

                     'Order Date']



print("1st time office produced higher sales than furniture is {}".format(first_time.date()))
# Timeseries model using Prophet 

    # analyze time-series with patterns of different time scales

        # e.g. yearly, weekly, daily 



from fbprophet import Prophet





# Fit models 



furniture = furniture.rename(columns = {'Order Date': 'ds', 

                                       'Sales': 'y'})

furniture_model = Prophet(interval_width = 0.95)

furniture_model.fit(furniture)



office = office.rename(columns = {'Order Date': 'ds', 

                                 'Sales': 'y'})

office_model = Prophet(interval_width = 0.95)

office_model.fit(office)
# Set model freq 

    # periods = integer number of periods to forecast forward 

    # 12 = forecast 1 year ahead, given freq = month



furniture_forecast = furniture_model.make_future_dataframe(periods = 12,

                                                          freq = 'MS')

furniture_forecast = furniture_model.predict(furniture_forecast)



office_forecast = office_model.make_future_dataframe(periods = 12,

                                                    freq = 'MS')

office_forecast = office_model.predict(office_forecast)

plt.figure(figsize=(18, 6))

furniture_model.plot(furniture_forecast,

                    xlabel = 'Date', ylabel = 'Sales')



plt.title('Furniture Sales');
plt.figure(figsize = (18, 6))

office_model.plot(office_forecast,

                 xlabel = 'Date', ylabel = 'Sales')



plt.title('Office sales');
# Compare forecasts 



# modulus operator, gives remainder of values after dividing 

furniture_names = ['furniture_%s' % column for column in furniture_forecast.columns]

office_names = ['office_%s' % column for column in furniture_forecast.columns]



# Prepare dataframes for merge 

merge_furniture_forecast = furniture_forecast.copy()

merge_office_forecast = office_forecast.copy()



merge_furniture_forecast.columns = furniture_names

merge_office_forecast.columns = office_names 



# Order date = ds

# rough merge 

forecast = pd.merge(merge_furniture_forecast, 

                   merge_office_forecast,

                   how = 'inner',

                   left_on = 'furniture_ds',

                   right_on = 'office_ds')



# correct dates 

forecast = forecast.rename(columns = {'furniture_ds': 'Date'}).drop('office_ds',axis = 1)



forecast.head()
# Visualize trend & forecast



plt.figure(figsize = (10, 7))

plt.plot(forecast['Date'], 

        forecast['furniture_trend'],

        'b-')

plt.plot(forecast['Date'],

        forecast['office_trend'],

        'r-')



plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')

plt.title('Furniture v Office');



# Sales growth trend
plt.figure(figsize=(10, 7))



plt.plot(forecast['Date'], 

         forecast['furniture_yhat'], 

         'b-')



plt.plot(forecast['Date'],

        forecast['office_yhat'], 'r-')



plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')



plt.title('Sale estimate')
# Use Prophet models to inspect different trends



furniture_model.plot_components(furniture_forecast);
office_model.plot_components(office_forecast);
# linear increase over time, 



# office growth > furniture 