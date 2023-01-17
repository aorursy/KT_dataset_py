import numpy as np  # Scientific computing

import pandas as pd # Data analysis and manipulation



from datetime import datetime # Dates and times manipulations



# Visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Plotly is a graphing library for interactive, publication-quality graphs

# pip install plotly==4.5.4

import plotly.graph_objects as go                   

import plotly.express as px

from plotly.subplots import make_subplots



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore")
# Loading the PJM East energy consumption raw data

df_raw = pd.read_csv('/kaggle/input/hourly-energy-consumption/PJME_hourly.csv', index_col = 'Datetime', parse_dates = ['Datetime'])
# Creating a data cleaning function

def data_clean(df):

    

    # Sorting the datetime index

    df.sort_index(inplace = True)

    

    # Dropping datetime duplicates

    df = df[~df.index.duplicated()]

    

    # Setting the frequence to be Hourly

    df = df.asfreq('H')

    

    # Renaming the PJME_MW column to energy

    df.rename(columns = {'PJME_MW' : 'energy'}, inplace = True)

    

    # Filling the Missing values using the preceding values

    df.energy = df.energy.fillna(method = 'ffill')

    

    return df
# Creating a clean dataframe from the raw data

df_clean = data_clean(df_raw)

df_clean.head()
df_clean.info()
df_clean.describe()
# Creating a function to extract some features

def data_prep(df):

    df['year'] = df.index.year

    df['month'] = df.index.month

    df['month_name'] = df.index.month_name()

    df['week_of_year'] = df.index.weekofyear

    df['quarter'] = df.index.quarter

    df['day_of_week'] = df.index.dayofweek

    df['day_of_week_name'] = df.index.day_name()

    df['day_of_month'] = df.index.day

    df['day_of_year'] = df.index.dayofyear

    df['hour'] = df.index.hour

    

    return df
# Adding useful features to the cleaned dataframe

df = data_prep(df_clean)

df.head()
# Creating times grouped dataframes in order to analyse them

df_year = df.groupby('year')['energy'].sum()

df_month = df.groupby('month_name', sort = False)['energy'].sum()

df_week_of_year = df.groupby('week_of_year')['energy'].sum()

df_quarter = df.groupby('quarter')['energy'].sum()

df_day_of_week = df.groupby('day_of_week_name', sort = False)['energy'].sum()

df_day_of_month = df.groupby('day_of_month')['energy'].sum()

df_day_of_year = df.groupby('day_of_year')['energy'].sum()

df_hour = df.groupby('hour')['energy'].sum()
# Creating a plotly subplot

fig = make_subplots(rows=4, cols=2, vertical_spacing = 0.175,

                    subplot_titles=(['Year', 'Month', 'Day of Week', 'Day of Month', 'Week of Year', 'Day of year', 'Hour', 'Quarter']))



fig.add_trace(go.Scatter(x=df_year.index, y = df_year), row=1, col=1)

fig.add_trace(go.Scatter(x=df_month.index, y=df_month), row=1, col=2)

fig.add_trace(go.Scatter(x=df_week_of_year.index, y=df_week_of_year), row=3, col=1)

fig.add_trace(go.Scatter(x=df_day_of_week.index, y=df_day_of_week), row=2, col=1)

fig.add_trace(go.Scatter(x=df_day_of_month.index, y=df_day_of_month), row=2, col=2)

fig.add_trace(go.Scatter(x=df_day_of_year.index, y=df_day_of_year), row=3, col=2)

fig.add_trace(go.Scatter(x=df_hour.index, y=df_hour), row=4, col=1)

fig.add_trace(go.Scatter(x=df_quarter.index, y=df_quarter), row=4, col=2)



fig.update_layout(title = 'Energy Consumption of PJME per', height = 700, showlegend = False)



fig.show()
# Plotting the time series

fig = px.line(df, x=df.index, y = df.energy)



fig.update_layout(title = "Energy Consumption of PJME from 2002 to 2018", 

                  yaxis_title="Energy (MW)", 

                  xaxis_title="Date", 

                  xaxis_rangeslider_visible=True)

fig.show()
from statsmodels.tsa.seasonal import seasonal_decompose
# Since our time series has an hourly frequence, we can set the seasonal decomposition frequence to

# 24 * 365 = 8760 to capture any Yearly seasonality

# 24 * 30.5 = 732 to capture any Montly seasonality

# And so on ....

year_period = int(24 * 365)

month_period = int(24 * 30.5)



s_dec_additive = seasonal_decompose(df.energy, freq = year_period, model = 'additive')

s_dec_multiplicative = seasonal_decompose(df.energy, freq = year_period, model = 'multiplicative')



s_dec_additive_monthly = seasonal_decompose(df[:year_period].energy, freq = month_period, model = 'additive')

s_dec_multiplicative_monthly = seasonal_decompose(df[:year_period].energy, freq = month_period, model = 'multiplicative')
# Plotting the components

def plot_seasonal(res, axes, model):

    axes[0].set_title(model)

    res.observed.plot(ax=axes[0], legend=False)

    axes[0].set_ylabel('Observed')

    res.trend.plot(ax=axes[1], legend=False)

    axes[1].set_ylabel('Trend')

    res.seasonal.plot(ax=axes[2], legend=False)

    axes[2].set_ylabel('Seasonal')

    res.resid.plot(ax=axes[3], legend=False)

    axes[3].set_ylabel('Residual')
# Plotting the yearly seasonal decompose

fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(12,5))



plot_seasonal(s_dec_additive, axes[:,0], 'Additive')

plot_seasonal(s_dec_multiplicative, axes[:,1], 'Multiplicative')



plt.tight_layout()

plt.show()
# Plotting the monthly seasonal decompose

fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(12,5))



plot_seasonal(s_dec_additive_monthly, axes[:,0], 'Additive')

plot_seasonal(s_dec_multiplicative_monthly, axes[:,1], 'Multiplicative')



plt.tight_layout()

plt.show()
import statsmodels.tsa.stattools as sts
# Running the Adfuller test

adfuller = sts.adfuller(df.energy)



# Extracting the test and critical values

test_value, critical_values = adfuller[0], adfuller[4]
print(f'The test value : \t{test_value}\nThe critical values : \t{critical_values}')
# Creating a function to split the data into train and test based on a given size

def data_split(df, size):

    return df[:size], df[size:]
# Setting up the size

size = int(0.8 * len(df))



# Creating the train and test dataframes

train, test = data_split(df_clean, size)



train_hours = len(train)

test_hours = len(test)



print(f'{str(train.energy.tail())} \n\n {str(test.energy.head())}')
# Plotting the dataframes

fig = make_subplots()



fig.add_trace(go.Line(x = train.index, y = train.energy, name = "Train") )



fig.add_trace(go.Line(x = test.index, y = test.energy, name = 'Test'))



fig.update_layout(title = "Energy Consumption of PJME / Train and Test", 

                  yaxis_title="Energy (MW)", 

                  xaxis_title="Date", xaxis_rangeslider_visible=True)



fig.show()
from fbprophet import Prophet
# Creating a function to prepare our data

def data_prep_prophet(df):

    return df.reset_index().copy().rename(columns = {'Datetime' : 'ds', 'energy':'y'})[['ds', 'y']]
train_p = data_prep_prophet(train)

train_p.head()
# Instantiating a new Prophet object with Multiple Seasonalities

model_p = Prophet(yearly_seasonality=True)

model_p.add_seasonality(name='monthly', period = month_period, fourier_order=5)



# Fitting the model to the train dataframe

model_p.fit(train_p);
# Setting up the period of prediction to be the length of the test Hours in the future

future_p = model_p.make_future_dataframe(periods = test_hours, freq = 'H')



# Creating a forecast dataframe that contains the forecasted values - yhat 

forecast_p = model_p.predict(future_p)



forecast_p[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Saving the forcast into a csv file

forecast_p.to_csv('forecast_p.csv')
fig, ax = plt.subplots(1, figsize=(15,5))

model_p.plot(forecast_p, ax = ax);
model_p.plot_components(forecast_p);
# Creating a funtion to Plot the test with the forecast results

def plot_forecast_test(test, forecast, model):

    fig = make_subplots()



    fig.add_trace(go.Line(x=test.index, y=test.energy, name = 'Test'))

    fig.add_trace(go.Line(x=forecast.ds, y = forecast.yhat, name = "Forecast") )



    fig.update_layout(title = "Energy Consumption of PJME / Forecast VS Test / {}".format(model), 

                      yaxis_title="Energy (MW)", 

                      xaxis_title="Date", 

                      xaxis=dict(

                          rangeselector=dict(

                              buttons=list([

                                  dict(count=1, label="1m", step ="month", stepmode="backward"),

                                  dict(count=6, label="6m", step ="month", stepmode="backward"),

                                  dict(count=1, label="1y", step ="year", stepmode="backward"),

                                  dict(step="all")

                              ])

                          ),

                          rangeslider=dict(visible=True),

                          type="date")

                     )

    return fig.show()
plot_forecast_test(test, forecast_p[train_hours:], 'Prophet')
import xgboost as xgb
# Defining the features

features = ['year', 'month', 'week_of_year', 'quarter', 'day_of_week', 'day_of_month', 'day_of_year', 'hour']



X_train, y_train = train[features], train['energy']

X_test, y_test = test[features], test['energy']
# Fitting XGB regressor 

model_xgb = xgb.XGBRegressor()

model_xgb.fit(X_train, y_train)

print(model_xgb)
# Making the prediction on the X_test set 

future_xgb = model_xgb.predict(data = X_test)



# Creating the forecast dataframe

forecast_xgb = pd.DataFrame()

forecast_xgb['ds'] = X_test.index

forecast_xgb["yhat"] = future_xgb



forecast_xgb.tail()
# Saving the forcast into a csv file

forecast_xgb.to_csv("forecast_xgb.csv")
plot_forecast_test(test, forecast_xgb, 'XGBoost')
from sklearn.metrics import mean_squared_error, mean_absolute_error
def mean_absolute_percentage_error(y_true, y_pred):

    """Calculates MAPE given y_true and y_pred"""

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# Creating an error metrics printing function

def print_err(y_true, y_pred):

    print('MSE   :\t', round(mean_squared_error(y_true, y_pred), 3))

    print('RMSE  :\t', round(np.sqrt(mean_squared_error(y_true, y_pred)), 3))

    print('MAE   :\t', round(mean_absolute_error(y_true, y_pred), 3))

    print('MAPE  :\t', round(mean_absolute_percentage_error(y_true, y_pred), 3), '%')
y_true = test.energy

y_pred_p = forecast_p[train_hours:].yhat

y_pred_xgb = forecast_xgb.yhat
print_err(y_true, y_pred_p)
print_err(y_true, y_pred_xgb)