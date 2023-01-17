import seaborn as sns

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

usecols = ['dt_iso', 'temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 

           'clouds_all', 'weather_main', 'weather_description']

df = pd.read_csv('../input/openweatherdata-denpasar-1990-2020v0.1.csv', parse_dates=True, index_col='dt_iso', usecols=usecols)

df['date'] = df.index

df = df[['date', 'temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 

           'clouds_all', 'weather_main', 'weather_description']]

df.info()
df.head(5)
print(df['weather_main'].value_counts())

plt.figure(1, figsize=(8, 6))

sns.countplot(y='weather_main', data=df)

plt.show()
print(df['weather_description'].value_counts())

plt.figure(1, figsize=(7,15))

sns.countplot(y='weather_description', data=df)

plt.show()
# Function to resampling time series data

def data_resample(data, time):

    """

    data: Dataframe

    time: Resampling frequencies

    """

    if time == 'hourly':

        data = data.resample('H').mean() # hour

    elif time == 'daily':

        data = data.resample('D').mean() # day

    elif time == 'weekly':

        data = data.resample('W').mean() # week

    elif time == 'monthly':

        data = data.resample('M').mean() # month

    elif time == 'quarterly':

        data = data.resample('Q').mean() # quarter

    elif time == 'yearly':

        data = data.resample('A').mean() # year

    

    return data
def line_plot(data, plot_kind, xlabel, title):

    plt.figure(1, figsize=(12, 5))

    data.plot(kind=plot_kind)

    plt.xlabel(xlabel)

    plt.title(title)

    plt.show()
line_plot(data_resample(df.loc['1990':'2019']['temp'], 'yearly'), 'line', 'year', 'Yearly Temperature Data')

line_plot(data_resample(df.loc['1990':'2019']['temp'], 'monthly'), 'line', 'year', 'Montly Temperature Data')

line_plot(data_resample(df.loc['1990':'2019']['temp'], 'weekly'), 'line', 'year', 'Weekly Temperature Data')
from scipy import stats



index = 0

dt_col = ['temp', 'pressure', 'humidity', 'wind_speed']

label = ['Temperature', 'Pressure', 'Humidity', 'Wind Speed']

plt.figure(1, figsize=(15,9))

for subplot in range(221, 225):

    plt.subplot(subplot)

    sns.distplot(df[dt_col[index]], kde=False, fit=stats.gamma)

    plt.xlabel(label[index], fontweight='bold')

    index += 1

plt.tight_layout()    

plt.show()
index = 0

weather_col = ['wind_deg', 'clouds_all']

plt.figure(1, figsize=(15,9))

for subplot in range(221, 223):

    plt.subplot(subplot)

    sns.distplot(df[weather_col[index]], kde=False, fit=stats.gamma)

    plt.title(weather_col[index], fontweight='bold')

    index += 1

    

plt.show()
dt = df['wind_deg'].value_counts()

dt = pd.DataFrame(dt)

dt = dt.reset_index()

dt = dt.rename(columns={'index': 'wind_deg', 'wind_deg': 'num'})



N = len(dt)

width = np.pi / 4 * np.random.rand(N)

colors = plt.cm.viridis(radii / 10.)



ax = plt.subplot(111, projection='polar')

ax.bar(x=dt['wind_deg'], height=dt['num'], width=width, bottom=0.0, color=colors, alpha=0.5)

plt.figure(1, figsize=(15,9))

plt.show()
# Check for stationarity

plt.figure(1, figsize=(15,6))

df['temp'].plot() # plot hourly data for 20 years
index = 0

resample_time = ['yearly', 'monthly', 'weekly', 'daily']

plt.figure(1, figsize=(15,9))

for subplot in range(221, 225):

    plt.subplot(subplot)

    dt = data_resample(df['temp'], resample_time[index])

    plt.plot(dt)

    plt.title(resample_time[index], fontweight='bold')

    index = index + 1

    

plt.show()
df_temp = data_resample(df['temp'].loc['1990':'2019'], 'daily')

df_temp.head()
from statsmodels.tsa.stattools import adfuller



def adf_test(timeseries):

    print('Results of Dickey-Fuller Test:')

    df_test = adfuller(timeseries, autolag='AIC')

    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in df_test[4].items():

        df_output['Critical Value (%s)' % key] = value

    print(df_output)

    

# Apply ADF Test

adf_test(df_temp)
from statsmodels.tsa.stattools import kpss



def kpss_test(timeseries):

    print('Results of KPSS Test:')

    kpss_test = kpss(timeseries, regression='c')

    kpss_output = pd.Series(kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lag Used'])

    for key, value in kpss_test[3].items():

        kpss_output['Critical Value (%s)' % key] = value

        

    print(kpss_output)

    

# Apply KPSS Test

kpss_test(df_temp)
df_diff = df_temp - df_temp.shift(1)

df_temp['diff'] = df_diff

df_temp['diff'].dropna().plot()
def test_stationarity(timeseries, info):

    """

    timeseries: pandas time series

    info: resampling information, (hourly, daily, weekly, monthly, quarterly, yearly)

    """

    

    # Determining rolling statistics

    windowsize = 24

    rolmean = timeseries.rolling(windowsize).mean()

    rolstd = timeseries.rolling(windowsize).mean()

    

    # Plot rolling statistics:

    orig = plt.plot(timeseries, color='yellow', label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label='Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation ' + info)

    plt.show(block=False)

    

    # Perform Dickey-Fuller Test:

    adf_test(timeseries)

    

    print('\n')

    

    # Perform KPSS Test:

    kpss_test(timeseries)
df_temp = data_resample(df.loc['1990':'2019']['temp'], 'hourly')

test_stationarity(df_temp, 'Monthly')