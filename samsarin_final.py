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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import plotly.graph_objects as go

%matplotlib inline



df = pd.read_excel('/kaggle/input/anzar-data/Data - Macro data and price reaction.xlsx')



df.head()
indicator = []

value = []

for i in range(len(df['Indicator'].value_counts())):

  indicator.append(df['Indicator'].value_counts().index[i])

  value.append(df['Indicator'].value_counts()[i])



value = (np.array(value)*100/sum(value))
fig = go.Figure([go.Bar(x=indicator, y=value)])

fig.show()
sns.countplot(df['Performance'])
df.isna().sum()
df.describe()
actuals = []

for i in range(len(df)):

  if df.iloc[i]['Actuals'] == '  ':

    actuals.append(0)

  else:

    actuals.append(df.iloc[i]['Actuals'])



df['Actuals'] = actuals
error = []

dates = []

for date in df['Date'].unique():

  dates.append(str(date).split('T')[0])

  error.append(sum(df[df['Date']==date]['Actuals']-df[df['Date']==date]['Forecast']))
fig = go.Figure()

fig.add_trace(go.Scatter(x=dates, y=error,

                    mode='lines+markers',

                    name='lines+markers'))
df_timeseries = pd.read_excel('/kaggle/input/anzar-data/Data - Macro data and price reaction.xlsx',sheet_name='Financial assets data',parse_dates=[0])
new_header = df_timeseries.iloc[0]

df_timeseries = df_timeseries[1:]



df_timeseries.columns = new_header

df_timeseries.drop('USD_TWI',inplace = True,axis = 1)

df_timeseries.head()
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA
plt.plot(df_timeseries['Date'],df_timeseries['S&P_500'])
rolling_mean = df_timeseries['S&P_500'].rolling(window = 12).mean()

rolling_std = df_timeseries['S&P_500'].rolling(window = 12).std()

plt.plot(df_timeseries['Date'],df_timeseries['S&P_500'], color = 'blue', label = 'Original')

plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')

plt.plot(rolling_std, color = 'black', label = 'Rolling Std')

plt.legend(loc = 'best')

plt.title('Rolling Mean & Rolling Standard Deviation')

plt.show()
result = adfuller(df_timeseries['S&P_500'])

print('ADF Statistic: {}'.format(result[0]))

print('p-value: {}'.format(result[1]))

print('Critical Values:')

for key, value in result[4].items():

    print('\t{}: {}'.format(key, value))
def get_stationarity(timeseries):

    

    # rolling statistics

    rolling_mean = timeseries.rolling(window=12).mean()

    rolling_std = timeseries.rolling(window=12).std()

    

    # rolling statistics plot

    original = plt.plot(timeseries, color='blue', label='Original')

    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')

    std = plt.plot(rolling_std, color='black', label='Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    # Dickey–Fuller test:

    result = adfuller(timeseries)

    print('ADF Statistic: {}'.format(result[0]))

    print('p-value: {}'.format(result[1]))

    print('Critical Values:')

    for key, value in result[4].items():

        print('\t{}: {}'.format(key, value))
df_log = np.log(df_timeseries['S&P_500'].astype(float))

plt.plot(df['Date'],df_log)
rolling_mean = df_log.rolling(window=12).mean()

df_log_minus_mean = df_log - rolling_mean

df_log_minus_mean.dropna(inplace=True)

get_stationarity(df_log_minus_mean)
df_log = pd.DataFrame(df_log)

df_log['Date'] = df_timeseries['Date']
decomposition = seasonal_decompose(df_log['S&P_500'],period=30) 

model = ARIMA(df_log['S&P_500'], order=(2,1,2))

results = model.fit(disp=-1)

decomposition.plot()