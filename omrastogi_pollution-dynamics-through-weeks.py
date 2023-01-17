import numpy as np 
import pandas as pd 
import seaborn as sns 
from matplotlib import pyplot as plt
%matplotlib inline
pol_df = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
item_df = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')  
print (pol_df.info())
print (item_df.info())
pol_df.head()
pol_df[pol_df.SO2 < 0].head()
pol_df = pol_df[pol_df.SO2 > 0]
pol_df[pol_df.SO2 < 0].head()
pol_df['year'] = pol_df['Measurement date'].str.split('-').str.get(0)
pol_df['month'] = pol_df['Measurement date'].str.split('-').str.get(1)
pol_df['datetime'] = pol_df['Measurement date'].str.split('-').str.get(2)
pol_df['date'] = pol_df['datetime'].str.split(' ').str.get(0) 
pol_df['time'] = pol_df['datetime'].str.split(' ').str.get(1)
pol_df['hour'] = pol_df['time'].str.split(':').str.get(0)
pol_df = pol_df.drop(columns = ['time','datetime'])
pol_df = pol_df.drop(columns = ['Address'])
pol_df.head(3)
# week = pd.Timestamp(year = 2017,  month = 11 , day = 23).weekday()
pol_df['Day'] = pol_df['Measurement date'] = pd.to_datetime(pol_df['Measurement date'])

pol_df['Day'] = pol_df['Day'].dt.weekday
pol_df['Date'] = pol_df['Measurement date'].dt.date
measure = pol_df.drop(columns= ['Measurement date', 'hour', 'date', 'month'])
measure.head()
measure_weekly = pd.DataFrame(columns = ['Date' ,'Latitude', 'Longitude','SO2',  'NO2', 'CO', 'PM10', 'PM2.5', 'Day'])
codes = pd.unique(measure['Station code'])
dates = pd.unique(measure['Date'])
grouped = measure.groupby(by = 'Station code')
arr1 = []
arr2 = []
for code in codes:
    dbf = grouped.get_group(code)
    mean1 = dbf.groupby(by = 'Date').mean()
    arr1.append(mean1)
    mean2 = dbf.groupby(by = 'Day').mean()
    arr2.append(mean2)
weekday_dict = {
    0:'Monday',
    1:'Tuesday',
    2:'Wednesday',
    3:'Thursday',
    4:'Friday',
    5:'Saturday',
    6:'Sunday'
}            
everyday = pd.concat(arr1,ignore_index=False)
weekday_avg = pd.concat(arr2,ignore_index=False)

# everyday['Day'] = everyday['Day'].replace(weekday_dict)

everyday.head()
everyday.head(10)
station_avg = measure.groupby(by = 'Station code').mean()
station_avg = station_avg.drop(columns = ['Day'])
station_avg.head(5)
station_avg['Code'] = codes
plt.figure(figsize = [15,8])
sns.barplot(x= 'Code', y='SO2' , data = station_avg)
plt.plot()
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,12), sharex=True, constrained_layout=True)
fig.suptitle('Pollutant concentrations NO2, CO2, and O3 along the week', fontsize=16, fontweight='bold')
sns.lineplot(x='Day', y='SO2', data= everyday, ax=ax1)
sns.lineplot(x='Day', y='NO2', data=everyday, ax=ax2)
sns.lineplot(x='Day', y='O3', data= everyday, ax= ax3)
plt.plot()
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,12), sharex=True, constrained_layout=True)
fig.suptitle('Pollutant concentrations of CO, PM10 and PM2.5 along the week', fontsize=16, fontweight='bold')
sns.lineplot(x='Day', y='CO', data= everyday, ax= ax1)
sns.lineplot(x='Day', y='PM10', data=everyday, ax= ax2)
sns.lineplot(x='Day', y='PM2.5', data=everyday, ax= ax3)
plt.plot()