# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DATA_FILE= '../input/uber-raw-data-sep14.csv'

uber_data = pd.read_csv(DATA_FILE)

uber_data.head()
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format="%m/%d/%Y %H:%M:%S")

uber_data['DayOfWeekNum'] = uber_data['Date/Time'].dt.dayofweek

uber_data['DayOfWeek'] = uber_data['Date/Time'].dt.weekday_name

uber_data['MonthDayNum'] = uber_data['Date/Time'].dt.day

uber_data['HourOfDay'] = uber_data['Date/Time'].dt.hour

uber_data['MinOfDay'] = uber_data['Date/Time'].dt.minute

uber_data.head()
weekday = uber_data.pivot_table(index=['DayOfWeek'],

                                  values='Base',

                                  aggfunc='count')

weekday.head()
weekdayAverage= weekday/30

weekdayAverage.head()
weekdayAverage.plot(kind='bar')

plt.ylabel('Average Rides Per Day')

plt.title('Average Rides per Day vs Day of Week')
uber_saturday = uber_data.loc[uber_data['DayOfWeek'] == "Saturday"]

uber_saturday.head()
Hours = uber_saturday.pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

Hours/30

Hours.plot(kind='bar')

plt.ylabel('Number of Rides')

plt.title('Number of Rides vs Hour of Day')
import folium



print('Folium installed and imported!')
#uber_monthDAY_map = uber_data.loc[uber_data['MonthDayNum'] == 6]

#uber_monthDAY_map.shape
uber_weekDAY_map = uber_data.loc[uber_data['DayOfWeek'] == "Friday"]

uber_weekDAY_map.shape
uber_TIME_map = uber_weekDAY_map.loc[uber_weekDAY_map['HourOfDay'] == 0]

uber_TIME_map.shape
limit = 1000 #if you just need a test, limit the data that you will input in map

uber_LIMIT_map = uber_TIME_map.iloc[0:limit, :]

uber_LIMIT_map.shape
ny_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)



# add markers to map

for lat, lng in zip(uber_LIMIT_map['Lat'], uber_LIMIT_map['Lon']):

    

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(ny_map)  

    

ny_map
uber_LIMIT_map.head()
uber_data_Stat = uber_data.drop(columns=['Lat','Lon'])

plot_uber_data_Stat = uber_data_Stat.set_index(['Base'])

plot_uber_data_Stat.head()
plot_uber_data_Stat.reset_index(inplace=True)

plot_uber_data_Stat.head()
plot_Stats = plot_uber_data_Stat.pivot_table(index=['MonthDayNum'],

                                  values='Base',

                                  aggfunc='count')

plot_Stats.head()




plot_Stats.plot(kind='bar')

plt.ylabel('Number of Rides')

plt.title('Number of Rides vs Day of Month')