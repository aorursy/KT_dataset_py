# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

 # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd



from scipy import stats



# data visualization

import matplotlib.pyplot as plt

%matplotlib inline



# prophet by Facebook

from fbprophet import Prophet



# Accuracy - RMSE

from sklearn.metrics import mean_squared_error

from math import sqrt
# importing data

df = pd.read_csv("../input/air-passengers/AirPassengers.csv")

df.head(n=3)
from pandas.tseries.offsets import MonthEnd



df['Date'] = pd.to_datetime(df['Month'], format="%Y-%m") + MonthEnd(1)

df.head(n=3)
df = df.drop(columns=['Month'])

df.head(n=3)
df.dtypes
# # Date to datetime64

# df['Date'] = pd.DatetimeIndex(df['Date'])

# df.dtypes
# from the prophet documentation every variables should have specific names

df = df.rename(columns = {'Date': 'ds',

                                '#Passengers': 'y'})

df.head(n=3)
pd.plotting.register_matplotlib_converters()



# plot monthly passengers count

ax = df.set_index('ds').plot(figsize = (12, 4))

ax.set_ylabel('passengers count')

ax.set_xlabel('Month')

plt.show()
def make_comparison_dataframe(historical, forecast):

    """Join the history with the forecast.

    

       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.

    """

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
def calculate_forecast_errors(df, prediction_size):

    """Calculate MAPE and MAE of the forecast.

    

       Args:

           df: joined dataset with 'y' and 'yhat' columns.

           prediction_size: number of days at the end to predict.

    """

    

    # Make a copy

    df = df.copy()

    

    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.

    df['e'] = df['y'] - df['yhat']

    df['p'] = 100 * df['e'] / df['y']

    

    # Recall that we held out the values of the last `prediction_size` days

    # in order to predict them and measure the quality of the model. 

    

    # Now cut out the part of the data which we made our prediction for.

    predicted_part = df[-prediction_size:]

    

    # Define the function that averages absolute error values over the predicted part.

    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

    

    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.

    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}
def inverse_boxcox(y, lambda_):

    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
df1 = df.copy()

# df1.head(3)
print("Minimum airpassengers count :", df1['y'].min())

print("Maximum airpassengers count :", df1['y'].max())
df1['cap'] = 700

df1['floor'] = 100
df1.head(n=3)
# Train and Validation dataset

prediction_size = 12

train_df1 = df1[:-prediction_size]

train_df1.tail(n=3)
# Build Prophet model

m1 = Prophet(growth='logistic',seasonality_mode='multiplicative',interval_width=0.95,mcmc_samples=300)

m1.fit(train_df1)
# Forecast

future1 = m1.make_future_dataframe(periods=prediction_size, freq='M')



future1['cap'] = 700

future1['floor'] = 100



future1.tail(n=3)
forecast1 = m1.predict(future1)

# forecast1.tail(n=3)



cmp_df1 = make_comparison_dataframe(df, forecast1)

cmp_df1.tail(n=3)
m1.plot(forecast1);
m1.plot_components(forecast1);
for err_name, err_value in calculate_forecast_errors(cmp_df1, prediction_size).items():

    print(err_name, err_value)
rmse = sqrt(mean_squared_error(cmp_df1['y'], cmp_df1['yhat']))

print("RMSE : ",rmse)
# Original dataset

df2 = df.copy()

# df2.head(n=3)
# Train and Validation dataset

prediction_size = 12

train_df2 = df2[:-prediction_size]

train_df2.tail(n=3)
# Box Cox transformation - y values

train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])

train_df2.tail(n=3)
print("Minimum airpassengers count :", train_df2['y'].min())

print("Maximum airpassengers count :", train_df2['y'].max())
train_df2['cap'] = 11

train_df2['floor'] = 6
# plot monthly passengers count

ax = train_df2.set_index('ds').plot(figsize = (12, 4))

ax.set_ylabel('passengers count')

ax.set_xlabel('Month')

plt.show()
# Prophet model

m2 = Prophet(growth='logistic',interval_width=0.95,mcmc_samples=300)

m2.fit(train_df2)
# Forecast

future2 = m2.make_future_dataframe(periods=prediction_size, freq='M')



future2['cap'] = 11

future2['floor'] = 6



future2.tail(n=3)
forecast2 = m2.predict(future2)

# forecast2.tail(n=3)
m2.plot(forecast2);
for column in ['yhat', 'yhat_lower', 'yhat_upper']:

    forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)
cmp_df2 = make_comparison_dataframe(df2, forecast2)

cmp_df2.tail(n=3)
cmp_df2.head(n=3)
for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():

    print(err_name, err_value)
rmse = sqrt(mean_squared_error(cmp_df2['y'], cmp_df2['yhat']))

print("RMSE : ",rmse)