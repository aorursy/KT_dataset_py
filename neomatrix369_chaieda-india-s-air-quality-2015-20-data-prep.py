import os
import warnings
import numpy as np
import pandas as pd
from math import pi
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from IPython.display import HTML,display

sns.set(style="whitegrid", font_scale=1.75)


# prettify plots
plt.rcParams['figure.figsize'] = [20.0, 5.0]
    
%matplotlib inline

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
%%time
df_station_hour = pd.read_csv("/kaggle/input/air-quality-data-in-india/station_hour.csv")
df_city_hour    = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_hour.csv")
df_station_day  = pd.read_csv("/kaggle/input/air-quality-data-in-india/station_day.csv")
df_city_day     = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")
df_stations     = pd.read_csv("/kaggle/input/air-quality-data-in-india/stations.csv")
print('Below is a list of columns of tables just as they are loaded:')
print('~~~')
print(f'df_stations: {list(df_stations.columns)}')
print('~~~')
print(f'df_station_day: {list(df_station_day.columns)}')
print('~~~')
print(f'df_station_hour: {list(df_station_hour.columns)}')
print('~~~')
print(f'df_city_day: {list(df_city_day.columns)}')
print('~~~')
print(f'df_city_hour: {list(df_city_hour.columns)}')
print('~~~')
fields_to_show = ['City','AQI_Bucket']
fields_to_ignore = ['StationId', 'StationName', 'State', 'Status', 'Region', 'Month', 'Year', 'Season', 'City', 'Date', 'AQI', 'AQI_Bucket']
names_of_pollutants = list(set(df_city_day.columns) - set(fields_to_ignore))
print(f"Names of Pollutants: {list(names_of_pollutants)}")
%%time
df_station_day['AQI_Bucket'].fillna('Unknown', inplace=True)
df_station_hour['AQI_Bucket'].fillna('Unknown', inplace=True)
df_city_day['AQI_Bucket'].fillna('Unknown', inplace=True)
df_city_hour['AQI_Bucket'].fillna('Unknown', inplace=True)
regions = ['1. Northern', '2. North Eastern', '3. Central', '4. Eastern', '5. Western', '6. Southern']
state_to_region_mapping = {
    'Andhra Pradesh': regions[4], 'Assam': regions[1] , 'Bihar': regions[3], 'Chandigarh': regions[0],  
    'Delhi': regions[0], 'Gujarat': regions[4], 'Haryana': regions[0], 'Jharkhand': regions[3], 
    'Karnataka': regions[5], 'Kerala': regions[5], 'Madhya Pradesh': regions[2], 'Maharashtra': regions[5], 
    'Meghalaya': regions[1], 'Mizoram': regions[1], 'Odisha': regions[3], 'Punjab': regions[0], 
    'Rajasthan': regions[0], 'Tamil Nadu': regions[5], 'Telangana': regions[5], 'Uttar Pradesh': regions[0],
    'West Bengal': regions[3]
}

def state_to_region(state):
    if state in state_to_region_mapping:
        return state_to_region_mapping[state]
    return 'None'
%%time
df_stations['Region'] = df_stations['State'].apply(state_to_region)
df_stations
df_stations['Status'].fillna('Unknown', inplace=True)
df_stations
%%time
df_stations.to_feather('stations_transformed.fth')
%%time
df_city_day = df_city_day.merge(df_stations)
df_city_day[fields_to_show + list(df_stations.columns)]
%%time
df_station_day = df_station_day.merge(df_stations)
df_station_day[fields_to_show + list(df_stations.columns)]
%%time
df_city_hour = df_city_hour.merge(df_stations)
df_city_hour[fields_to_show + list(df_stations.columns)]
%%time
df_station_hour = df_station_hour.merge(df_stations)
df_station_hour[fields_to_show + list(df_stations.columns)]
old_and_new_fields_to_show = list(set(['Region', 'Season', 'Year', 'Month', 
                                       'Weekday_or_weekend', 'Regular_day_or_holiday', 'AQ_Acceptability'] + fields_to_show) 
                                  - set(['StationId', 'Date']))
# The country's meteorological department follows the international standard of four seasons with some local adjustments: 
# - winter (January and February)
# - summer (March, April and May) 
# - monsoon (rainy) season (June to September)
# - post-monsoon period (October to December)

date_to_season_mapping = {'1. Winter': [1, 2], '2. Summer': [3, 5], '3. Monsoon': [6, 9],  '4. Post-Monsoon': [10, 12]}

def date_to_season(dates):
    results = []
    date_values = dates.values
    for date in date_values:
        month = int(date.split('-')[1])
        result = 'None'
        for each_season in date_to_season_mapping:
            start, end = date_to_season_mapping[each_season]
            if ((start < end) and (start <= month <= end)) or \
               ((start > end) and ((month >= start) or (month <= end))):
                result = each_season
                break

        results.append(result)
    return results
month_no_to_name_mapping = [
    '01. Jan', '02. Feb', '03. Mar', '04. Apr', '05. May', '06. Jun', '07. Jul', 
    '08. Aug', '09. Sep', '10. Oct', '11. Nov', '12. Dec'
]

def date_to_month_name(dates):
    month_values = pd.DatetimeIndex(dates).month.values
    results = []
    for month in month_values:
        result = month_no_to_name_mapping[month - 1]
        results.append(result)
    return results

def weekday_or_weekend(dates):
    results = []
    for date_value in pd.DatetimeIndex(dates.values):
        weekno = date_value.weekday()
        result = "Weekday" if weekno < 5 else "Weekend"
        results.append(result)
    return results

import holidays
holidays_india = holidays.India()

def regular_day_or_holiday(dates):
    results = []
    for date_value in pd.DatetimeIndex(dates.values):
        result = "Holiday (or Festival)" if date_value.date() in holidays_india else "Regular day"
        results.append(result)
    return results
def aq_acceptability(aqi_bucket):
    results = []
    for each_aqi_bucket in aqi_bucket.values:
        result = "Acceptable" if each_aqi_bucket \
                in ["Good", "Satisfactory"] else "Unacceptable"
        results.append(result)
    return results
%%time
df_city_day['Month'] = date_to_month_name(df_city_day['Date'])
df_city_day['Year'] = pd.DatetimeIndex(df_city_day['Date']).year
df_city_day['Season'] = date_to_season(df_city_day['Date'])
df_city_day['Weekday_or_weekend'] = weekday_or_weekend(df_city_day['Date'])
df_city_day['Regular_day_or_holiday'] = regular_day_or_holiday(df_city_day['Date'])
df_city_day['AQ_Acceptability'] = aq_acceptability(df_city_day['AQI_Bucket'])
df_city_day[old_and_new_fields_to_show]
%%time
df_city_day.to_feather('city_day_transformed.fth')
%%time
df_station_day['Month'] = date_to_month_name(df_station_day['Date'])
df_station_day['Year'] = pd.DatetimeIndex(df_station_day['Date']).year
df_station_day['Season'] = date_to_season(df_station_day['Date'])
df_station_day['Weekday_or_weekend'] = weekday_or_weekend(df_station_day['Date'])
df_station_day['Regular_day_or_holiday'] = regular_day_or_holiday(df_station_day['Date'])
df_station_day['AQ_Acceptability'] = aq_acceptability(df_station_day['AQI_Bucket'])
df_station_day[old_and_new_fields_to_show]
%%time
df_station_day.to_feather('station_day_transformed.fth')
date_to_day_period_mapping = {'1. Morning': [4, 11], '2. Afternoon': [12, 17], 
                              '3. Evening': [18, 19], '4. Night': [20, 4]}
def date_to_day_period(datetimes):
    results = []
    datetime_values = datetimes.values
    for datetime in datetime_values:
        _, time_of_day = datetime.split(' ')
        hour, _, _ = time_of_day.split(':')
        hour = int(hour)
        result = 'None'
        for each_day_period in date_to_day_period_mapping:
            start, end = date_to_day_period_mapping[each_day_period]
            if ((start < end) and (start <= hour <= end)) or \
               ((start > end) and ((hour >= start) or (hour <= end))):
                result = each_day_period
                break

        results.append(result)
    return results
%%time
df_city_hour['Day_period'] = date_to_day_period(df_city_hour['Datetime'])
df_city_hour['Month'] = date_to_month_name(df_city_hour['Datetime'])
df_city_hour['Year'] = pd.DatetimeIndex(df_city_hour['Datetime']).year
df_city_hour['Season'] = date_to_season(df_city_hour['Datetime'])
df_city_hour['Weekday_or_weekend'] = weekday_or_weekend(df_city_hour['Datetime'])
df_city_hour['Regular_day_or_holiday'] = regular_day_or_holiday(df_city_hour['Datetime'])
df_city_hour['AQ_Acceptability'] = aq_acceptability(df_city_hour['AQI_Bucket'])
df_city_hour[set(old_and_new_fields_to_show + ["Day_period", "Weekday_or_weekend", 'Regular_day_or_holiday', 'AQ_Acceptability'])]
%%time
df_city_hour.to_feather('city_hour_transformed.fth')
%%time
df_station_hour['Day_period'] = date_to_day_period(df_station_hour['Datetime'])
df_station_hour['Month'] = date_to_month_name(df_station_hour['Datetime'])
df_station_hour['Year'] = pd.DatetimeIndex(df_station_hour['Datetime']).year
df_station_hour['Season'] = date_to_season(df_station_hour['Datetime'])
df_station_hour['Weekday_or_weekend'] = weekday_or_weekend(df_station_hour['Datetime'])
df_station_hour['Regular_day_or_holiday'] = regular_day_or_holiday(df_station_hour['Datetime'])
df_station_hour['AQ_Acceptability'] = aq_acceptability(df_station_hour['AQI_Bucket'])
df_station_hour[set(old_and_new_fields_to_show + ["Day_period", "Weekday_or_weekend", 
                                                  'Regular_day_or_holiday', 'AQ_Acceptability'])]
%%time
df_station_hour.to_feather('station_hour_transformed.fth')