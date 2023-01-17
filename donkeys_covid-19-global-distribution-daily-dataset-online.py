

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import requests



URL = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"

r = requests.get(url = URL) 
from io import StringIO



csv_io = StringIO(r.text)

df = pd.read_csv(csv_io)
df.head()
df["dateRep"] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
df["countriesAndTerritories"].unique()
df.dtypes
df_norway = df[df["countriesAndTerritories"] == "Norway"].sort_values(by="dateRep")[["dateRep", "cases", "deaths"]]

df_sweden = df[df["countriesAndTerritories"] == "Sweden"].sort_values(by="dateRep")[["dateRep", "cases", "deaths"]]

df_finland = df[df["countriesAndTerritories"] == "Finland"].sort_values(by="dateRep")[["dateRep", "cases", "deaths"]]

df_norway.set_index('dateRep', inplace=True)

df_sweden.set_index('dateRep', inplace=True)

df_finland.set_index('dateRep', inplace=True)

import matplotlib.dates as mdates

import matplotlib.pyplot as plt

%matplotlib inline



def dual_plot(col_name, df, type_name):

    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(20,12))



    ax = plt.subplot(2, 2, 1)

    plt.xticks(rotation=45) #have to set rotate here before labels created

    locator = mdates.DayLocator(interval=15)

    ax.xaxis.set_major_locator(locator)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    plt.plot(df.index, df[col_name], 'c-')

    plt.title(f'{type_name}s per Day')

    plt.ylabel(f'{type_name}s')

    plt.xlabel('Date')



    ax = plt.subplot(2, 2, 2)

    plt.xticks(rotation=45) #have to set rotate here before labels created

    locator = mdates.DayLocator(interval=15)

    ax.xaxis.set_major_locator(locator)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    plt.plot(df.index, df[col_name].cumsum(), 'r-')

    plt.xlabel('Date')

    plt.ylabel(f'{type_name}s')

    plt.title(f'{type_name}s, cumulative')



    plt.show()
dual_plot("cases", df_norway, "Case")
dual_plot("cases", df_sweden, "Case")
dual_plot("cases", df_finland, "Case")
!ls /kaggle/input/country-to-continent
df_countries = pd.read_csv("/kaggle/input/country-to-continent/countryContinent.csv", encoding="iso-8859-1")

df_countries.head(10)
set(df["countryterritoryCode"]) - set(df_countries["code_3"])
df_countries.append({"country": "Kosovo", "code_2": "XK", "code_3": "XKX", "continent": "Europe", "sub_region": "Southern Europe", "region_code": 150, "sub_region_code": 39}, ignore_index=True)

pass
df[df["countryterritoryCode"] == "XKX"].head()
df_m = pd.merge(df, df_countries, left_on='countryterritoryCode', right_on='code_3')

df_m.head(20)
continents = df_m["continent"].unique()

continents
#df_sorted_cumulative = pd.DataFrame()

#for country in countries:

#    df_country = df_m[df_m["country"] == country].sort_values(by="dateRep")[["country", "continent", "sub_region", "dateRep", "cases", "deaths"]]

#    df_country["cum_cases"] = df_country["cases"].cumsum()

#    df_country["cum_deaths"] = df_country["deaths"].cumsum()

#    df_sorted_cumulative = pd.concat([df_sorted_cumulative, df_country], axis=0)

#df_sorted_cumulative.head(100)
df_sorted_cumulative = pd.DataFrame()

for continent in continents:

    continent_countries = df_m[df_m["continent"] == continent]["country"].unique()

    for country in continent_countries:

        df_country = df_m[df_m["country"] == country].sort_values(by="dateRep")[["country", "continent", "sub_region", "dateRep", "cases", "deaths"]]

        df_country["cum_cases"] = df_country["cases"].cumsum()

        df_country["cum_deaths"] = df_country["deaths"].cumsum()

        df_sorted_cumulative = pd.concat([df_sorted_cumulative, df_country], axis=0)

df_sorted_cumulative.head(100)
df_total_cumulative = df_sorted_cumulative.groupby("dateRep").sum()
df_total_cumulative.tail()
df_sorted_cumulative_continent = pd.DataFrame()

for continent in continents:

    df_continent = df_sorted_cumulative[df_sorted_cumulative["continent"] == continent]

    df_continent = df_continent.sort_values(by="dateRep")

    df_continent["cum_cases"] = df_continent["cases"].cumsum()

    df_continent["cum_deaths"] = df_continent["deaths"].cumsum()

    df_sorted_cumulative_continent = pd.concat([df_sorted_cumulative_continent, df_continent], axis=0)

df_sorted_cumulative_continent.tail(10)
df_sorted_cumulative_continent[df_sorted_cumulative_continent["continent"] == "Asia"].tail()
continents
df_asia = df_sorted_cumulative_continent[df_sorted_cumulative_continent["continent"] == "Asia"].groupby("dateRep").max()

df_africa = df_sorted_cumulative_continent[df_sorted_cumulative_continent["continent"] == "Africa"].groupby("dateRep").max()

df_europe = df_sorted_cumulative_continent[df_sorted_cumulative_continent["continent"] == "Europe"].groupby("dateRep").max()

df_americas = df_sorted_cumulative_continent[df_sorted_cumulative_continent["continent"] == "Americas"].groupby("dateRep").max()

df_oceania = df_sorted_cumulative_continent[df_sorted_cumulative_continent["continent"] == "Oceania"].groupby("dateRep").max()
dual_plot("cases", df_total_cumulative, "Case")
dual_plot("deaths", df_total_cumulative, "Death")
dual_plot("cases", df_europe, "Case")
dual_plot("deaths", df_europe, "Death")
dual_plot("cases", df_americas, "Case")
dual_plot("deaths", df_americas, "Death")
dual_plot("cases", df_africa, "Case")
dual_plot("deaths", df_africa, "Death")
dual_plot("cases", df_oceania, "Case")
dual_plot("deaths", df_oceania, "Death")
#Possibly deeper splits

df_countries["sub_region"].unique()