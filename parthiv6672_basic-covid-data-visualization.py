# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import datetime

import operator 

import os 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
confirmed_global = pd.read_csv('../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

confirmed_global.head()
deaths_global = pd.read_csv('../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

deaths_global.head()
recovered_global = pd.read_csv('../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

recovered_global.head()
### Replace NaN values by 0

confirmed_global.fillna(0, inplace=True)

deaths_global.fillna(0, inplace=True)

recovered_global.fillna(0, inplace=True)
df_confirm = confirmed_global

df_confirm.shape

df_confirm.columns

df_confirm.head()
df_deaths = deaths_global

df_deaths.shape
df_recov = recovered_global

df_recov.shape
#df_latest = pd.read_csv('../input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports/06-18-2020.csv')

#df_latest.head()

#df_latest.shape
#df_latest_us = pd.read_csv('../input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports_us/06-18-2020.csv')

#df_latest_us.head()

#df_latest_us.shape
cols = df_recov.keys()

cols
# List "Confirmed" is now limited to values only. Province/State, Country etc not included. 



confirmed = df_confirm.loc[:, cols[4]:cols[-1]]

deaths = df_deaths.loc[:, cols[4]:cols[-1]]

recoveries = df_recov.loc[:, cols[4]:cols[-1]]

confirmed
# First we are pulling out the dates from the dataframe df_confirm or the list titled confirmed. Next we define the various lists we want to track.

# Next,we are calculating the total as of 6/30 the number of confirmed, death and recoveries globally

# Next, we are appending to the glbl_cases list, the sum of cases for each date in dates - ie. 161 values in the list



dates = confirmed.keys()

glbl_cases = []

total_deaths = [] 

mortality_rate = []

recovery_rate = [] 

total_recovered = [] 

active_cases = []

active_cases_sum = []



for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    glbl_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    total_recovered.append(recovered_sum)

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)

    active_cases.append(confirmed_sum-(recovered_sum+death_sum))

    active_cases_sum=(confirmed_sum-(recovered_sum+death_sum)).sum()



    

dates
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
plt.figure(figsize=(36, 16))

plt.style.use('ggplot')

plt.plot(days_since_1_22, glbl_cases, color='black')

plt.plot(days_since_1_22, total_recovered, color='green')

plt.plot(days_since_1_22, total_deaths, color='red')

plt.plot(days_since_1_22, active_cases, color='orange')

plt.title('Confirmed, Active, Recovery & Death Cases of Coronavirus Over Time', size=30)

plt.legend(['Number of Confirmed Cases= '+ str(confirmed_sum), 'Number of Recovery Cases='+ str(recovered_sum), 'Number of Death Cases='+ str(death_sum), 'Number of Active Cases= '+ str(active_cases_sum)], loc='upper left', fontsize=15)

plt.xlabel('Time in Days since 01/22', size=30)

plt.ylabel('Number of Cases', size=30)

plt.xticks(rotation=50, size=12)

plt.show()
china_cases = [] 

italy_cases = []

us_cases = [] 

spain_cases = [] 

mexico_cases = [] 

peru_cases = [] 

uk_cases = [] 

russia_cases = [] 

brazil_cases = []

india_cases = [] 



china_deaths = [] 

italy_deaths = []

us_deaths = [] 

spain_deaths = [] 

mexico_deaths = [] 

peru_deaths = [] 

uk_deaths = [] 

russia_deaths = []

brazil_deaths = [] 

india_deaths = []



china_recov = [] 

italy_recov = []

us_recov = [] 

spain_recov = [] 

mexico_recov = [] 

peru_recov = [] 

uk_recov = [] 

russia_recov = [] 

brazil_recov = [] 

india_recov = [] 





for i in dates:

    china_cases.append(df_confirm[df_confirm['Country/Region']=='China'][i].sum())

    italy_cases.append(df_confirm[df_confirm['Country/Region']=='Italy'][i].sum())

    us_cases.append(df_confirm[df_confirm['Country/Region']=='US'][i].sum())

    spain_cases.append(df_confirm[df_confirm['Country/Region']=='Spain'][i].sum())

    mexico_cases.append(df_confirm[df_confirm['Country/Region']=='Mexico'][i].sum())

    peru_cases.append(df_confirm[df_confirm['Country/Region']=='Peru'][i].sum())

    uk_cases.append(df_confirm[df_confirm['Country/Region']=='United Kingdom'][i].sum())

    russia_cases.append(df_confirm[df_confirm['Country/Region']=='Russia'][i].sum())

    brazil_cases.append(df_confirm[df_confirm['Country/Region']=='Brazil'][i].sum())

    india_cases.append(df_confirm[df_confirm['Country/Region']=='India'][i].sum())

    

    china_deaths.append(df_deaths[df_deaths['Country/Region']=='China'][i].sum())

    italy_deaths.append(df_deaths[df_deaths['Country/Region']=='Italy'][i].sum())

    us_deaths.append(df_deaths[df_deaths['Country/Region']=='US'][i].sum())

    spain_deaths.append(df_deaths[df_deaths['Country/Region']=='Spain'][i].sum())

    mexico_deaths.append(df_deaths[df_deaths['Country/Region']=='Mexico'][i].sum())

    peru_deaths.append(df_deaths[df_deaths['Country/Region']=='Peru'][i].sum())

    uk_deaths.append(df_deaths[df_deaths['Country/Region']=='United Kingdom'][i].sum())

    russia_deaths.append(df_deaths[df_deaths['Country/Region']=='Russia'][i].sum())

    brazil_deaths.append(df_deaths[df_deaths['Country/Region']=='Brazil'][i].sum())

    india_deaths.append(df_deaths[df_deaths['Country/Region']=='India'][i].sum())

    

    china_recov.append(df_recov[df_recov['Country/Region']=='China'][i].sum())

    italy_recov.append(df_recov[df_recov['Country/Region']=='Italy'][i].sum())

    us_recov.append(df_recov[df_recov['Country/Region']=='US'][i].sum())

    spain_recov.append(df_recov[df_recov['Country/Region']=='Spain'][i].sum())

    mexico_recov.append(df_recov[df_recov['Country/Region']=='Mexico'][i].sum())

    peru_recov.append(df_recov[df_recov['Country/Region']=='Peru'][i].sum())

    uk_recov.append(df_recov[df_recov['Country/Region']=='United Kingdom'][i].sum())

    russia_recov.append(df_recov[df_recov['Country/Region']=='Russia'][i].sum())

    brazil_recov.append(df_recov[df_recov['Country/Region']=='Brazil'][i].sum())

    india_recov.append(df_recov[df_recov['Country/Region']=='India'][i].sum())

    

#mexico_cases

#india_deaths

#china_recov

#us_recov
plt.figure(figsize=(36, 16))

plt.style.use('ggplot')

plt.plot(days_since_1_22, china_cases, color='black')

plt.plot(days_since_1_22, italy_cases, color='blue')

plt.plot(days_since_1_22, us_cases, color='red', linewidth=7.0)

plt.plot(days_since_1_22, spain_cases, color='green')

plt.plot(days_since_1_22, mexico_cases, color='yellow')

plt.plot(days_since_1_22, peru_cases, color='orange')

plt.plot(days_since_1_22, uk_cases, color='purple')

plt.plot(days_since_1_22, russia_cases, color='brown', linewidth=5.0)

plt.plot(days_since_1_22, brazil_cases, color='magenta', linewidth=5.0)

plt.plot(days_since_1_22, india_cases, color='pink', linewidth=7.0)

plt.title('# of Coronavirus Cases', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['China','Italy','US', 'Spain', 'Mexico', 'Peru',  'UK', 'Russia', 'Brazil', 'India'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(36, 16))

plt.style.use('ggplot')

plt.plot(days_since_1_22, china_deaths, color='black')

plt.plot(days_since_1_22, italy_deaths, color='blue')

plt.plot(days_since_1_22, us_deaths, color='red', linewidth=7.0)

plt.plot(days_since_1_22, spain_deaths, color='green')

plt.plot(days_since_1_22, mexico_deaths, color='yellow')

plt.plot(days_since_1_22, peru_deaths, color='orange')

plt.plot(days_since_1_22, uk_deaths, color='purple', linewidth=7.0)

plt.plot(days_since_1_22, russia_deaths, color='brown', linewidth=5.0)

plt.plot(days_since_1_22, brazil_deaths, color='magenta', linewidth=5.0)

plt.plot(days_since_1_22, india_deaths, color='pink', linewidth=7.0)

plt.title('# of Coronavirus Cases', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['China','Italy','US', 'Spain', 'Mexico', 'Peru',  'UK', 'Russia', 'Brazil', 'India'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()