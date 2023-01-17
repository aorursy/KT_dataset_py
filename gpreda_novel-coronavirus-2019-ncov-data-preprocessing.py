import numpy as np

import pandas as pd

import datetime as dt

data_df = pd.read_csv("..//input//novel-corona-virus-2019-dataset//covid_19_data.csv")
print(f"Data: rows: {data_df.shape[0]}, cols: {data_df.shape[1]}")

print(f"Data columns: {list(data_df.columns)}")



print(f"Days: {data_df.ObservationDate.nunique()} ({data_df.ObservationDate.min()} : {data_df.ObservationDate.max()})")

print(f"Country/Region: {data_df['Country/Region'].nunique()}")

print(f"Province/State: {data_df['Province/State'].nunique()}")

print(f"Confirmed all: {sum(data_df.groupby(['Province/State'])['Confirmed'].max())}")

print(f"Recovered all: {sum(data_df.loc[~data_df.Recovered.isna()].groupby(['Province/State'])['Recovered'].max())}")

print(f"Deaths all: {sum(data_df.loc[~data_df.Deaths.isna()].groupby(['Province/State'])['Deaths'].max())}")



print(f"Diagnosis: days since last update: {(dt.datetime.now() - dt.datetime.strptime(data_df.ObservationDate.max(), '%m/%d/%y')).days} ")
data_df.head()
data_df.info()
country_sorted = list(data_df['Country/Region'].unique())

country_sorted.sort()

print(country_sorted)
data_df.loc[data_df['Country/Region']=='Holy See', 'Country/Region'] = 'Vatican City'

data_df.loc[data_df['Country/Region']==' Azerbaijan', 'Country/Region'] = 'Azerbaijan'

data_df.loc[data_df['Country/Region']=='Republic of Ireland', 'Country/Region'] = 'Ireland'

data_df.loc[data_df['Country/Region']=="('St. Martin',)", 'Country/Region'] = 'St. Martin'
province_sorted = list(data_df.loc[~data_df['Province/State'].isna(), 'Province/State'].unique())

province_sorted.sort()

print(province_sorted)
diamond_list = list(data_df.loc[data_df['Province/State'].str.contains("Diamond", na=False), 'Province/State'].unique())

diamond_list.sort()

print(diamond_list)
province_ = list(data_df.loc[~data_df['Province/State'].isna(), 'Province/State'].unique())

country_ = list(data_df['Country/Region'].unique())



common_province_country = set(province_) & set(country_)

print(common_province_country)
for province in list(common_province_country):

    country_list = list(data_df.loc[data_df['Province/State']==province, 'Country/Region'].unique())

    print(province, country_list)
counties_us = list(data_df.loc[(~data_df['Province/State'].isna()) & \

                               data_df['Province/State'].str.contains("County,", na=False) &\

                               (data_df['Country/Region']=='US'), 'Province/State'].unique())

counties_us.sort()

print(counties_us)
cities_places_us = list(data_df.loc[(~data_df['Province/State'].isna()) & \

                               (~data_df['Province/State'].str.contains("County,", na=False)) &\

                               (data_df['Province/State'].str.contains(",", na=False)) &\

                               (data_df['Country/Region']=='US'), 'Province/State'].unique())

cities_places_us.sort()

print(cities_places_us)
states_us = list(data_df.loc[(~data_df['Province/State'].isna()) & \

                               (~data_df['Province/State'].str.contains("County,", na=False)) &\

                               (~data_df['Province/State'].str.contains(",", na=False)) &\

                               (data_df['Country/Region']=='US'), 'Province/State'].unique())

states_us.sort()

print(states_us)

print(len(states_us))
data_df.to_csv("covid_19_data.csv", index=False)