# Import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Set style & figures inline

sns.set()

%matplotlib inline
confirmed_cases_data_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

death_cases_data_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

recovery_cases_data_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
# Import data & check it out

raw_data_confirmed = pd.read_csv(confirmed_cases_data_url)

raw_data_confirmed.head()
# Group by region

data_day = raw_data_confirmed.groupby(['Country/Region']).sum().drop(['Lat', 'Long'], axis=1)

data_day.head()
df = data_day.transpose()
# Melt data so that it is long

data = data_day.reset_index().melt(id_vars='Country/Region', var_name='date')

data.head()
data.loc[(data.value < 1),'value'] = None

data.head()
# Pivot data to wide & index by date

df = data.pivot(index='date', columns='Country/Region', values='value')

df.tail()
# Set index as DateTimeIndex

datetime_index = pd.DatetimeIndex(df.index)

df.set_index(datetime_index, inplace=True)

df.head()
# Check out index

df.index
# Plot time series of several countries of interest

poi = ['China', 'US', 'Italy', 'France', 'Australia']

df[poi].plot(figsize=(20,10), linewidth=5, fontsize=20)

plt.xlabel('Date', fontsize=20);

plt.ylabel('Confirmed patients count', fontsize=20);

plt.title('Confirmed Patients Time Series', fontsize=20);
# Plot time series of several countries of interest

poi = ['China', 'US', 'Italy', 'France', 'Australia']

df[poi].plot(figsize=(20,10), linewidth=5, fontsize=20, logy=True)

plt.xlabel('Date', fontsize=20);

plt.ylabel('Confirmed Patients Logarithmic count', fontsize=20);

plt.title('Confirmed Patients Logarithmic Time Series', fontsize=20);
# Import data & check it out

raw_data_deaths = pd.read_csv(death_cases_data_url)

raw_data_deaths.head()
# Group by region

data_day = raw_data_deaths.groupby(['Country/Region']).sum().drop(['Lat', 'Long'], axis=1)

df = data_day.transpose()

# Melt data so that it is long

data = data_day.reset_index().melt(id_vars='Country/Region', var_name='date')

#

data.loc[(data.value < 25),'value'] = None

# Pivot data to wide & index by date

df = data.pivot(index='date', columns='Country/Region', values='value')

# Set index as DateTimeIndex

datetime_index = pd.DatetimeIndex(df.index)

df.set_index(datetime_index, inplace=True)
df.tail()
# Plot time series of several countries of interest

poi = ['China', 'US', 'Italy', 'France', 'Australia']

df[poi].plot(figsize=(20,10), linewidth=5, fontsize=20)

plt.xlabel('Date', fontsize=20);

plt.ylabel('Deaths Patients count', fontsize=20);

plt.title('Deaths Patients Time Series', fontsize=20);
df.dropna(axis=1, how='all', inplace=True)

df.head()
df = df.sort_index()

df1 = df.reset_index().drop(['date'], axis=1)

df1.head()
for col in df1.columns:

    print(col, df1[col].first_valid_index())

    df1[col] = df1[col].shift(-df1[col].first_valid_index())
df2 = df1.apply(lambda x: x.shift(-x.first_valid_index()))
# Plot time series of several countries of interest

df2.plot(figsize=(20,10), linewidth=5, fontsize=20)

plt.xlabel('Days', fontsize=20);

plt.ylabel('Deaths Patients count', fontsize=20);

plt.title('Deaths Patients Time Series', fontsize=20);
# Plot time series of several countries of interest

df2.plot(figsize=(20,10), linewidth=5, fontsize=20, logy=True)

plt.xlabel('Days', fontsize=20);

plt.ylabel('Deaths Patients Logarithmic count', fontsize=20);

plt.title('Deaths Patients Logarithmic Time Series', fontsize=20);
# Function for grouping countries by region

def grouping_by_region(raw_data, min_val):

    data_day = raw_data.groupby(['Country/Region']).sum().drop(['Lat', 'Long'], axis=1)

    df_t = data_day.transpose()

    # Melt data so that it is long

    data = data_day.reset_index().melt(id_vars='Country/Region', var_name='date')

    #

    data.loc[(data.value < min_val),'value'] = None

    # Pivot data to wide & index by date

    df_t = data.pivot(index='date', columns='Country/Region', values='value')

    # Set index as DateTimeIndex

    datetime_index = pd.DatetimeIndex(df_t.index)

    df_t.set_index(datetime_index, inplace=True)

    return df_t
# Function to plot time series of several countries of interest

def plot_time_series(df, plot_title, x_label, y_label, isLogY=False):

    df.plot(figsize=(20,10), linewidth=5, fontsize=20, logy=isLogY)

    plt.xlabel(x_label, fontsize=20);

    plt.ylabel(y_label, fontsize=20);

    plt.title(plot_title, fontsize=20);
# Function to manipulate the data

def data_manipulation(df):

    df.dropna(axis=1, how='all', inplace=True)

    df = df.sort_index()

    df1 = df.reset_index().drop(['date'], axis=1)

    

    for col in df1.columns:

        print(col, df1[col].first_valid_index())

        df1[col] = df1[col].shift(-df1[col].first_valid_index())

        

    df2 = df1.apply(lambda x: x.shift(-x.first_valid_index()))

    return df2
# Import data & check it out

raw_data_recovered = pd.read_csv(recovery_cases_data_url)

raw_data_recovered.head()
df = grouping_by_region(raw_data_recovered, 50)

df.tail()
# Plot time series of several countries of interest

poi = ['China', 'US', 'Italy', 'France', 'Australia']

plot_time_series(df[poi], 'Recovered Patients Time Series', 'Date', 'Recovered Patients count', False)
clean_df = data_manipulation(df)
clean_df.head()
plot_time_series(clean_df, 'Recovered Patients Time Series', 'Days', 'Recovered Patients count', False)
plot_time_series(clean_df, 'Recovered Patients Logarithmic Time Series', 'Days', 'Recovered Patients Logarithmic count', True)