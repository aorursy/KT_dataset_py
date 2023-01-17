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
import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt



%matplotlib inline



DATAPATH = 'https://raw.githubusercontent.com/marcopeix/air-quality/master/data/AirQualityUCI.csv'



data = pd.read_csv(DATAPATH, sep=';')

data.head()
# Make dates actual dates

data['Date'] = pd.to_datetime(data['Date'])



# Convert measurements to floats

for col in data.iloc[:,2:].columns:

    if data[col].dtypes == object:

        data[col] = data[col].str.replace(',', '.').astype('float')



# Compute the average considering only the positive values

def positive_average(num):

    return num[num > -200].mean()

    

# Aggregate data

daily_data = data.drop('Time', axis=1).groupby('Date').apply(positive_average)



# Drop columns with more than 8 NaN

daily_data = daily_data.iloc[:,(daily_data.isna().sum() <= 8).values]



# Remove rows containing NaN values

daily_data = daily_data.dropna()



# Aggregate data by week

weekly_data = daily_data.resample('W').mean()



# Plot the weekly concentration of each gas

def plot_data(col):

    plt.figure(figsize=(17, 8))

    plt.plot(weekly_data[col])

    plt.xlabel('Time')

    plt.ylabel(col)

    plt.grid(False)

    plt.show()

    

for col in weekly_data.columns:

    plot_data(col)
# Drop irrelevant columns

cols_to_drop = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']



weekly_data = weekly_data.drop(cols_to_drop, axis=1)



# Import Prophet

from fbprophet import Prophet

import logging



logging.getLogger().setLevel(logging.ERROR)



# Change the column names according to Prophet's guidelines

df = weekly_data.reset_index()

df.columns = ['ds', 'y']

df.head()



# Split into a train/test set

prediction_size = 30

train_df = df[:-prediction_size]



# Initialize and train a model

m = Prophet()

m.fit(train_df)



# Make predictions

future = m.make_future_dataframe(periods=prediction_size)

forecast = m.predict(future)

forecast.head()



# Plot forecast

m.plot(forecast)



# Plot forecast's components

m.plot_components(forecast)



# Evaluate the model

def make_comparison_dataframe(historical, forecast):

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))



cmp_df = make_comparison_dataframe(df, forecast)

cmp_df.head()



def calculate_forecast_errors(df, prediction_size):

    

    df = df.copy()

    

    df['e'] = df['y'] - df['yhat']

    df['p'] = 100 * df['e'] / df['y']

    

    predicted_part = df[-prediction_size:]

    

    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

    

    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}



for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():

    print(err_name, err_value)



# Plot forecast with upper and lower bounds

plt.figure(figsize=(17, 8))

plt.plot(cmp_df['yhat'])

plt.plot(cmp_df['yhat_lower'])

plt.plot(cmp_df['yhat_upper'])

plt.plot(cmp_df['y'])

plt.xlabel('Time')

plt.ylabel('Average Weekly NOx Concentration')

plt.grid(False)

plt.show()