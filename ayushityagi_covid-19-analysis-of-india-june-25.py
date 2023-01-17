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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
from plotly import tools

import plotly.express as px
complete_data = pd.read_csv('../input/covid-19-till-june25-india/complete.csv')

complete_data.head()
complete_data.info()
complete_data['Date'] = pd.to_datetime(complete_data['Date'])

complete_data['Date'] = complete_data['Date'].dt.strftime('%Y/%m/%d')
complete_data.rename(columns={'Name of State / UT':'State/UT','Total Confirmed cases':'Confirmed',

                              'Cured/Discharged/Migrated':'Recovered'},inplace=True)
data = complete_data[['Date','State/UT','Confirmed','Death','Recovered']]

data.head()
data.tail()
data.isnull().sum()
# getting number of active cases

data['Active'] = data['Confirmed'] - data['Death'] - data['Recovered']

data.head()
latest = data[data['Date'] == data['Date'].max()]

latest.head()
dates = data.groupby('Date')[['Confirmed','Death','Recovered','Active']].sum().reset_index()

dates.head()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.plot(dates.Date, dates.Confirmed, color='blue', label='Confirmed')

plt.plot(dates.Date, dates.Death, color='red', label='Deaths')

plt.plot(dates.Date, dates.Recovered, color='green', label='Recovered')

plt.fill_between(dates.Date, dates.Confirmed, color='lightblue', alpha=0.5)

plt.fill_between(dates.Date, dates.Death, color='orange', alpha=0.5)

plt.fill_between(dates.Date, dates.Recovered, color='lightgreen', alpha=0.5)

plt.xticks(dates.Date[::4],rotation='45')

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases', size=15)

plt.title('Corona cases over time in India', size=15)

plt.legend()

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.plot(dates.Date, dates.Active, color='orange', label='Active cases')

plt.plot(dates.Date, dates.Recovered, color='green', label='Recovered')

plt.fill_between(dates.Date, dates.Active, color='yellow', alpha=0.5)

plt.fill_between(dates.Date, dates.Recovered, color='lightgreen', alpha=0.5)

plt.xticks(dates.Date[::4],rotation='45')

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases', size=15)

plt.title('Active and Recovered cases over time', size=15)

plt.legend()

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.plot(dates.Date, dates.Active, color='green', label='Active cases')

plt.plot(dates.Date, dates.Death, color='red', label='Deaths')

plt.fill_between(dates.Date, dates.Active, color='lightgreen', alpha=0.5)

plt.fill_between(dates.Date, dates.Death, color='orange', alpha=0.5)

plt.xticks(dates.Date[::4],rotation='45')

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases', size=15)

plt.title('Active and Death cases over time', size=15)

plt.legend()

plt.show()
recovery_rate = np.round(dates['Recovered']/dates['Confirmed'], 3)*100

mortality_rate = np.round(dates['Death']/dates['Confirmed'], 3)*100



plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.plot(dates.Date, recovery_rate, color='orange', label='Recovered to per 100 confirmed cases')

plt.plot(dates.Date, mortality_rate, color='green', label='Deaths to per 100 confirmed cases')

plt.fill_between(dates.Date, recovery_rate, color='yellow', alpha=0.5)

plt.fill_between(dates.Date, mortality_rate, color='lightgreen', alpha=0.5)

plt.xticks(dates.Date[::4],rotation='45')

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases to per 100 cases', size=15)

plt.title('Recovery and mortality rate cases over time', size=15)

plt.legend()

plt.show()
dates['New_cases'] = 0

for i in dates.index-1:

    dates['New_cases'].iloc[i] = dates['Confirmed'].iloc[i]-dates['Confirmed'].iloc[i-1]

dates['New_cases'].iloc[0] = dates['Confirmed'].iloc[0]

dates.head()
plt.figure(figsize=(20,8))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

sns.barplot(dates.Date, dates.New_cases, palette='winter', edgecolor='k')

plt.xticks(rotation='90', size=7)

plt.title('Daily increase in cases', size=15)

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases', size=15)

plt.show()
temp = data.groupby(['Date','State/UT'])[['Confirmed','Death','Recovered','Active']].sum().reset_index()

temp.head()
temp['size'] = temp['Confirmed'].pow(0.3) * 3.5

temp['Latitude'] = complete_data['Latitude']

temp['Longitude'] = complete_data['Longitude']

px.scatter_geo(temp, lat='Latitude', lon='Longitude', locationmode='country names', color='Confirmed',

               hover_name='State/UT', size='size', range_color=[1,100], scope='asia', animation_frame='Date', 

               projection='natural earth', color_continuous_scale='jet', title='Covid-19 cases over time in India').show()
latest_data = latest.groupby('State/UT')['Confirmed','Death','Recovered','Active'].sum().reset_index()

latest_data.head()
top20_states = latest_data.sort_values('Confirmed',ascending=False).head(20).reset_index()

top20_states = top20_states.drop('index', axis=1)

top20_states
plt.figure(figsize=(14,8))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.plot(top20_states['State/UT'],top20_states['Confirmed'], 'bo-', label='Confirmed')

plt.plot(top20_states['State/UT'],top20_states['Recovered'], 'go-', label='Recovered')

plt.plot(top20_states['State/UT'],top20_states['Death'], 'ro-', label='Death')

plt.fill_between(top20_states['State/UT'],top20_states['Confirmed'], color='lightblue', alpha=0.5)

plt.fill_between(top20_states['State/UT'],top20_states['Recovered'], color='lightgreen', alpha=0.5)

plt.fill_between(top20_states['State/UT'],top20_states['Death'], color='orange', alpha=0.5)

plt.title('Covid19 with top 20 states in India', size=15)

plt.xlabel('State/UT name', size=15)

plt.xticks(rotation='90')

plt.ylabel('Number of cases', size=15)

plt.legend()

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.barh(top20_states['State/UT'],top20_states['Confirmed'], edgecolor='b')

plt.title('Confirmed cases of top 20 State/UT in India', size=15)

plt.xlabel('Confirmed cases', size=15)

plt.ylabel('State/UT', size=15)

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.barh(top20_states['State/UT'],top20_states['Death'], edgecolor='b')

plt.title('Death cases of top 20 State/UT in India', size=15)

plt.xlabel('Death cases', size=15)

plt.ylabel('State/UT', size=15)

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.barh(top20_states['State/UT'],top20_states['Recovered'], edgecolor='b')

plt.title('Recovered cases of top 20 Sate/UT in India', size=15)

plt.xlabel('Recovered cases', size=15)

plt.ylabel('State/UT', size=15)

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.barh(top20_states['State/UT'],top20_states['Active'], edgecolor='b')

plt.title('Active cases of top 20 State/UT in India', size=15)

plt.xlabel('Active cases', size=15)

plt.ylabel('State/UT', size=15)

plt.show()
Maharashtra = temp[temp['State/UT'] == 'Maharashtra']

Delhi = temp[temp['State/UT'] == 'Delhi']

Tamil_Nadu = temp[temp['State/UT'] == 'Tamil Nadu']
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.plot(Maharashtra['Date'], Maharashtra['Confirmed'], color='red', label='Cases in Maharashtra')

plt.plot(Delhi['Date'], Delhi['Confirmed'], color='green', label='Cases in Delhi')

plt.plot(Tamil_Nadu['Date'], Tamil_Nadu['Confirmed'], color='blue', label='Cases in Tamil Nadu')

plt.fill_between(Maharashtra['Date'], Maharashtra['Confirmed'], color='orange', alpha=0.3)

plt.fill_between(Delhi['Date'], Delhi['Confirmed'], color='lightgreen', alpha=0.3)

plt.fill_between(Tamil_Nadu['Date'], Tamil_Nadu['Confirmed'], color='lightblue', alpha=0.3)

plt.xlabel('Dates', size=15)

plt.xticks(rotation=90, size=7)

plt.ylabel('Confirmed cases', size=15)

plt.title('Covid19 spread over time')

plt.legend()

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.88')

plt.plot(Maharashtra['Date'], Maharashtra['Death'], color='red', label='Deaths in Maharashtra')

plt.plot(Delhi['Date'], Delhi['Death'], color='green', label='Deaths in Delhi')

plt.plot(Tamil_Nadu['Date'], Tamil_Nadu['Death'], color='blue', label='Deaths in Tamil_Nadu')

plt.fill_between(Maharashtra['Date'], Maharashtra['Death'], color='orange', alpha=0.5)

plt.fill_between(Delhi['Date'], Delhi['Death'], color='lightgreen', alpha=0.5)

plt.fill_between(Tamil_Nadu['Date'], Tamil_Nadu['Death'], color='lightblue', alpha=0.5)

plt.xlabel('Dates', size=15)

plt.xticks(rotation=90, size=7)

plt.ylabel('Count of deaths', size=15)

plt.title('Covid19 Deaths over time', size=15)

plt.legend()

plt.show()