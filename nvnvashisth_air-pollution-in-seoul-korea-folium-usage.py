# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import folium

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
measure_df = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

serverity_df = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')


def severitySO2(x):

    severity = ""

    if(x <= 0.02):

        severity = "Good"

    elif((x > 0.02) & (x <= 0.05)):

        severity = "Normal"

    elif((x > 0.05) & (x <= 0.15)):

        severity = "Bad"

    elif((x > 0.15) & (x <= 1.0)):

        severity = "Very Bad"

    return severity



def severityNO2(x):

    severity = ""

    if(x <= 0.03):

        severity = "Good"

    elif((x > 0.03) & (x <= 0.06)):

        severity = "Normal"

    elif((x > 0.06) & (x <= 0.2)):

        severity = "Bad"

    elif((x > 0.2) & (x <= 2.0)):

        severity = "Very Bad"

    return severity

    

def severityCO(x):

    severity = ""

    if(x <= 2):

        severity = "Good"

    elif((x > 2) & (x <= 9)):

        severity = "Normal"

    elif((x > 9) & (x <= 15)):

        severity = "Bad"

    elif((x > 15) & (x <= 50)):

        severity = "Very Bad"

    return severity

    

def severityO3(x):

    severity = ""

    if(x <= 0.03):

        severity = "Good"

    elif((x > 0.03) & (x <= 0.09)):

        severity = "Normal"

    elif((x > 0.09) & (x <= 0.15)):

        severity = "Bad"

    elif((x > 0.15) & (x <= 0.5)):

        severity = "Very Bad"

    return severity



def severityPM10(x):

    severity = ""

    if(x <= 30):

        severity = "Good"

    elif((x > 30) & (x <= 80)):

        severity = "Normal"

    elif((x > 80) & (x <= 150)):

        severity = "Bad"

    elif((x > 150) & (x <= 600)):

        severity = "Very Bad"

    return severity

    

def severityPM25(x):

    severity = ""

    if(x <= 15):

        severity = "Good"

    elif((x > 15) & (x <= 35)):

        severity = "Normal"

    elif((x > 35) & (x <= 75)):

        severity = "Bad"

    elif((x > 75) & (x <= 500)):

        severity = "Very Bad"

    return severity
date_time = measure_df['Measurement date'].str.split(" ", n=1, expand=True)

measure_df['date'] = date_time[0]

measure_df['time'] = date_time[1]

measure_df = measure_df.drop(['Measurement date'], axis=1)
measure_df[["SO2","NO2","O3","CO","PM10","PM2.5"]].describe()
print("-1 Values in all columns")

print("Total Rows : ", measure_df.shape)

print("SO2   : ",measure_df[measure_df.SO2 == -1].shape[0])

print("NO2   : ",measure_df[measure_df.NO2 == -1].shape[0])

print("CO    : ",measure_df[measure_df.CO == -1].shape[0])

print("O3    : ",measure_df[measure_df.O3 == -1].shape[0])

print("PM10  : ",measure_df[measure_df.PM10 == -1].shape[0])

print("PM2.5 : ",measure_df[measure_df['PM2.5'] == -1].shape[0])

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=-1, strategy='mean')

df_imputed = pd.DataFrame(imp.fit_transform(measure_df[["SO2","NO2","O3","CO","PM10","PM2.5"]]))

df_imputed.columns = measure_df[["SO2","NO2","O3","CO","PM10","PM2.5"]].columns

df_imputed.index = measure_df.index

remain_df = measure_df[measure_df.columns.difference(["SO2","NO2","O3","CO","PM10","PM2.5"])]

df = pd.concat([remain_df, df_imputed], axis=1)

df.head()
df['SO2 Severity'] = df.apply(lambda row: severitySO2(row['SO2']), axis=1)

df['NO2 Severity'] = df.apply(lambda row: severityNO2(row['NO2']), axis=1)

df['CO Severity'] = df.apply(lambda row: severityCO(row['CO']), axis=1)

df['O3 Severity'] = df.apply(lambda row: severityO3(row['O3']), axis=1)

df['PM10 Severity'] = df.apply(lambda row: severityPM10(row['PM10']), axis=1)

df['PM2.5 Severity'] = df.apply(lambda row: severityPM25(row['PM2.5']), axis=1)
df.head()
df_mean_date = df.groupby(['date'], as_index=False).agg({'SO2':'mean', 'NO2':'mean', 'O3':'mean', 'CO':'mean', 'PM10':'mean', 'PM2.5':'mean'})

df_mean_date['date'] = pd.to_datetime(df_mean_date.date)

df_mean_date.head()
plt.figure(figsize=(50,10)) 

sns.lineplot(data=df_mean_date, x='date',y='SO2')
plt.figure(figsize=(50,10)) 

sns.lineplot(data=df_mean_date, x='date',y='NO2')
plt.figure(figsize=(50,10)) 

sns.lineplot(data=df_mean_date, x='date',y='CO')
plt.figure(figsize=(50,10)) 

sns.lineplot(data=df_mean_date, x='date',y='O3')
plt.figure(figsize=(50,10)) 

sns.lineplot(data=df_mean_date, x='date',y='PM10')
plt.figure(figsize=(50,10)) 

sns.lineplot(data=df_mean_date, x='date',y='PM2.5')
plt.figure(figsize = (20,8))        

sns.heatmap(df[["SO2","NO2","O3","CO","PM10","PM2.5"]].corr(),annot=True, cmap = 'coolwarm')
main_df = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

execept_date_df = df[df.columns.difference(["date","time"])]

main_df['Measurement date'] = pd.to_datetime(main_df['Measurement date'])

main_df = pd.concat([main_df['Measurement date'], execept_date_df], axis=1)

main_df['hour'] = main_df['Measurement date'].apply(lambda x: x.hour)

main_df['month'] = main_df['Measurement date'].apply(lambda x: x.month)

main_df['day'] = main_df['Measurement date'].apply(lambda x: x.day)

main_df['week'] = main_df['Measurement date'].apply(lambda x: x.week)

main_df['year'] = main_df['Measurement date'].apply(lambda x: x.year)

main_df.head()
main_df['month'].unique()
main_df_2017 = main_df.loc[main_df['year'] == 2017]

main_df_2018 = main_df.loc[main_df['year'] == 2018]

main_df_2019 = main_df.loc[main_df['year'] == 2019]
from folium.plugins import HeatMap

def generateBaseMap(default_location=[37.572016, 127.005007], default_zoom_start=12):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map

# base_map = generateBaseMap()

# HeatMap(data=main_df[['Latitude', 'Longitude', 'PM2.5']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)

# base_map
df_year_list = []

for year in main_df.year.sort_values().unique():

    df_year_list.append(main_df.loc[main_df.year == year, ['Latitude', 'Longitude', 'PM2.5']].groupby(['Latitude', 'Longitude']).mean().reset_index().values.tolist())

from folium.plugins import HeatMapWithTime

base_map = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_year_list, radius=70, gradient={0.05: 'blue', 0.5: 'green', 0.75: 'yellow', 1.0: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map)

base_map
df_month_list_2017 = []

for month in main_df_2017.month.sort_values().unique():

    df_month_list_2017.append(main_df_2017.loc[main_df_2017.month == month, ['Latitude', 'Longitude', 'PM2.5']].groupby(['Latitude', 'Longitude']).mean().reset_index().values.tolist())

from folium.plugins import HeatMapWithTime

base_map = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_month_list_2017, radius=70, gradient={0.05: 'blue', 0.5: 'green', 0.75: 'yellow', 1.0: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map)

base_map
df_week_list_2017 = []

for week in main_df_2017.week.sort_values().unique():

    df_week_list_2017.append(main_df_2017.loc[main_df_2017.week == week, ['Latitude', 'Longitude', 'PM2.5']].groupby(['Latitude', 'Longitude']).mean().reset_index().values.tolist())

from folium.plugins import HeatMapWithTime

base_map = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_week_list_2017, radius=50, gradient={0.05: 'blue', 0.5: 'green', 0.75: 'yellow', 1.0: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map)

base_map