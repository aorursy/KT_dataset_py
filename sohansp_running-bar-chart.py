# Import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Set style & figures inline

sns.set()

%matplotlib inline
given_series = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

given_series.head(10)
# from datetime import datetime

given_series.drop(['Sno','Time','ConfirmedIndianNational','ConfirmedForeignNational'], axis=1, inplace=True)

given_series.drop(['Cured','Deaths'], axis=1, inplace=True)

given_series.rename(columns = {'State/UnionTerritory':'States'}, inplace=True)

# given_series.Date = [datetime.strptime(date, '%d/%m/%y').date() for date in given_series.Date]

given_series.Date
states = given_series.States.unique()

req_dates = given_series.Date.unique()

req_dates
time_series = pd.DataFrame(columns = [req_dates])

time_series['States'] = states

time_series = pd.DataFrame(time_series)

time_series = time_series.fillna(0)
for index,data in given_series.iterrows():

#     print(data)

#     print(time_series.loc[time_series.eq(data.States).any(1) == True,[data.Date]] )

    time_series.loc[time_series.eq(data.States).any(1) == True,data.Date] = data.Confirmed
time_series.columns
time_series.drop(['States'], axis=1, inplace=True)

time_series = time_series.set_index(states) 

time_series.head(10)
time_series = time_series.transpose()

time_series.drop(['Telengana'], axis=1, inplace=True)

time_series.head()
# from datetime import datetime

# date_index = pd.to_datetime(time_series.index)

# # date_index = time_series.index.to_pydatetime().date()

# # time_series.set_index(date_index, inplace=True)

# time_series.index
# State having the highest confirmed cases in India

time_series.idxmax(axis=1)[-1]
# Number of confirmed cases in the state having highest confirmed cases in USA

sorted(time_series.max(), reverse = True)[0]
ax = time_series.plot(figsize=(20,10), linewidth=5, marker='.', colormap='brg', fontsize=20);

ax.legend(ncol=3, loc='upper left')

plt.xlabel('Days', fontsize=20);

plt.ylabel('Number of Confirmed Cases', fontsize=20);

plt.title('Total number of reported coronavirus cases in Indian states', fontsize=20);
latest_counts = time_series.iloc[-1]

top_10_states = latest_counts.sort_values(ascending=False)[:10]

top_10_states.index
ax = time_series[top_10_states.index].plot(figsize=(20,10), linewidth=5, marker='.', colormap='brg', fontsize=20);

ax.legend(ncol=1, loc='upper left')

plt.xlabel('Days', fontsize=20);

plt.ylabel('Number of Confirmed Cases', fontsize=20);

plt.title('Total number of reported coronavirus cases in top 10 Indian states', fontsize=20);
!pip install bar_chart_race

import bar_chart_race as bcr
bcr.bar_chart_race(

    df=time_series,

    filename='covid-19_india.mp4',

    orientation='h',

    sort='desc',

    label_bars=True,

    n_bars=10,

#     use_index=True,

    steps_per_period=30,

    period_length=500,

    figsize=(6.5, 3.5),

    cmap='dark24',

    title='COVID-19 Confirmed Cases In India by States',

    bar_label_size=7,

    tick_label_size=7,

    filter_column_colors = True,

#     period_label_size=16,

    fig=None)