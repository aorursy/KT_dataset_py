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
# Additional Imports 

# essential libraries
import math
import random
from datetime import timedelta
from IPython.core.display import HTML
#import googlemaps
from datetime import datetime



# storing and anaysis
import numpy as np
import pandas as pd


%matplotlib inline



# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium

# converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters() 

#offline plotting 
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
from scipy import signal

def get2deriv(df, index, column, value):
    #pivot table of 2nd derive (growth factor)
    newdf = df.pivot_table(
        index=index, columns=column, values=value, aggfunc=np.sum
    ).fillna(method="ffill").fillna(0)
    # Growth factor: (delta Number_n) / (delta Number_n)
    #newdf['shift1'] = newdf.shift(1, axis=0)
    #newdf['shift2'] = newdf.shift(2, axis=0)
    newdf =  (newdf-newdf.shift(1, axis=0))/(newdf.shift(1, axis=0)-newdf.shift(2, axis=0))
    #newdf = newdf.drop(['shift1', 'shift2', 'value'], axis=1)
    
    newdf = newdf.replace(np.inf, np.nan).fillna(1.0)
    # Rolling mean (window: 7 days)
    newdf = newdf.rolling(3).mean().dropna().loc[:df[index].max(), :]
    # round: 0.01
    growth_value_df = newdf.round(2)
    #growth_value_df = smoothergf(growth_value_df, 0.5, 5)
    growth_value_df.tail()
    
    frame = growth_value_df.copy()
    N, K = frame.shape
    data = {'2ndderiv': frame.to_numpy().ravel('F'),
            column: np.asarray(frame.columns).repeat(N),
            index: np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=[index, column, '2ndderiv'])

def get2derivworld(df, index, value):
    #pivot table of 2nd derive (growth factor)
    newdf = df.pivot_table(
        index=index, values=value, aggfunc=np.sum
    ).fillna(method="ffill").fillna(0)
    # Growth factor: (delta Number_n) / (delta Number_n)
    #newdf['shift1'] = newdf.shift(1, axis=0)
    #newdf['shift2'] = newdf.shift(2, axis=0)
    newdf =  (newdf-newdf.shift(1, axis=0))/(newdf.shift(1, axis=0)-newdf.shift(2, axis=0))
    #newdf = newdf.drop(['shift1', 'shift2', 'value'], axis=1)
    
    newdf = newdf.replace(np.inf, np.nan).fillna(1.0)
    # Rolling mean (window: 7 days)
    newdf = newdf.rolling(3).mean().dropna().loc[:df[index].max(), :]
    # round: 0.01
    growth_value_df = newdf.round(2)
    #growth_value_df = smoothergf(growth_value_df, 0.5, 5)
    growth_value_df.tail()
    
    frame = growth_value_df.copy()
    N, K = frame.shape
    data = {'2ndderiv': frame.to_numpy().ravel('F'),
            index: np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=[index, '2ndderiv'])

def get1deriv(df, index, column, value):
    #pivot table of 2nd derive (growth factor)
    newdf = df.pivot_table(
        index=index, columns=column, values=value, aggfunc="sum"
    ).fillna(method="ffill").fillna(0)
    # Growth factor: (delta Number_n) / (delta Number_n)
    newdf = newdf.diff()
    newdf = newdf.replace(np.inf, np.nan).fillna(1.0)
    # Rolling mean (window: 7 days)
    newdf = newdf.rolling(3).mean().dropna().loc[:df[index].max(), :]
    # round: 0.01
    growth_value_df = newdf.round(2)
    #growth_value_df = smoothergf(growth_value_df, 0.5, 3)
    growth_value_df.tail()
    
    frame = growth_value_df.copy()
    N, K = frame.shape
    data = {'1stderiv': frame.to_numpy().ravel('F'),
            column: np.asarray(frame.columns).repeat(N),
            index: np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=[index, column,'1stderiv'])

def get1derivworld(df, index, value):
    #pivot table of 2nd derive (growth factor)
    newdf = df.pivot_table(
        index=index,values=value, aggfunc="sum"
    ).fillna(method="ffill").fillna(0)
    # Growth factor: (delta Number_n) / (delta Number_n)
    newdf = newdf.diff()
    newdf = newdf.replace(np.inf, np.nan).fillna(1.0)
    # Rolling mean (window: 7 days)
    newdf = newdf.rolling(3).mean().dropna().loc[:df[index].max(), :]
    # round: 0.01
    growth_value_df = newdf.round(2)
    #growth_value_df = smoothergf(growth_value_df, 0.5, 3)
    growth_value_df.tail()
    
    frame = growth_value_df.copy()
    N, K = frame.shape
    data = {'1stderiv': frame.to_numpy().ravel('F'),
            index: np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=[index, '1stderiv'])

def getdaily(df, index, column, value):
    #pivot table of 2nd derive (growth factor)
    newdf = df.pivot_table(
        index=index, columns=column, values=value, aggfunc="sum"
    ).fillna(method="ffill").fillna(0)
    # Growth factor: (delta Number_n) / (delta Number_n)
    newdf = newdf.diff()
    newdf = newdf.replace(np.inf, np.nan).fillna(1.0)
    # Rolling mean (window: 7 days)
    newdf = newdf.rolling(3).mean().dropna().loc[:df[index].max(), :]
    # round: 0.01
    growth_value_df = newdf.round(2)
    growth_value_df.tail()
    
    frame = growth_value_df.copy()
    N, K = frame.shape
    data = {'daily_case': frame.to_numpy().ravel('F'),
            column: np.asarray(frame.columns).repeat(N),
            index: np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=[index, column,'daily_case'])

def getlag(df, index, column, value):
    #pivot table of 2nd derive (growth factor)
    newdf = df.pivot_table(
        index=index, columns=column, values=value, aggfunc=np.sum
    ).fillna(method="ffill").fillna(0)
    # Growth factor: (delta Number_n) / (delta Number_n)
    newdf = newdf.shift(-7, axis=0)
    
    newdf = newdf.replace(np.inf, np.nan).fillna(0.0)
    # Rolling mean (window: 7 days)
    #newdf = newdf.rolling(3).mean().dropna().loc[:df[index].max(), :]
    # round: 0.01
    growth_value_df = newdf
    
    growth_value_df.tail()
    
    frame = growth_value_df.copy()
    N, K = frame.shape
    data = {'lag': frame.to_numpy().ravel('F'),
            column: np.asarray(frame.columns).repeat(N),
            index: np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=[index, column,'lag'])

def getlagdeath(df, index, column, value):
    #pivot table of 2nd derive (growth factor)
    newdf = df.pivot_table(
        index=index, columns=column, values=value, aggfunc=np.sum
    ).fillna(method="ffill").fillna(0)
    # Growth factor: (delta Number_n) / (delta Number_n)
    newdf = newdf.shift(-7, axis=0)
    
    newdf = newdf.replace(np.inf, np.nan).fillna(0.0)
    # Rolling mean (window: 7 days)
    #newdf = newdf.rolling(3).mean().dropna().loc[:df[index].max(), :]
    # round: 0.01
    growth_value_df = newdf
    
    growth_value_df.tail()
    
    frame = growth_value_df.copy()
    N, K = frame.shape
    data = {'lag_death': frame.to_numpy().ravel('F'),
            column: np.asarray(frame.columns).repeat(N),
            index: np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=[index, column,'lag_death'])

#global measure
def add_daily_measures(df):
    df.loc[0,'Daily Cases'] = df.loc[0,'ConfirmedCases']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']
    for i in range(1,len(df)):
        df.loc[i,'Daily Cases'] = df.loc[i,'ConfirmedCases'] - df.loc[i-1,'ConfirmedCases']
        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    return df

#smoothens out data for plotting (one line)
def smoother(df, index, col):
    noise = 2 * np.random.random(len(df[index])) - 1 # uniformly distributed between -1 and 1
    y_noise = df[col] + noise
    y_col = signal.savgol_filter(y_noise, 53, 3)
    return y_col

#smoothens growth factor 
def smoothergf(inputdata,w,imax):
    data = 1.0*inputdata
    data = data.replace(np.nan,1)
    data = data.replace(np.inf,1)
    #print(data)
    smoothed = 1.0*data
    normalization = 1
    for i in range(-imax,imax+1):
        if i==0:
            continue
        smoothed += (w**abs(i))*data.shift(i,axis=0)
        normalization += w**abs(i)
    smoothed /= normalization
    return smoothed


################## only for treemap ###############

class country_utils():
    def __init__(self):
        self.d = {}
    
    def get_dic(self):
        return self.d
    
    def get_country_details(self,country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            if country_obj is None:
                c = pycountry.countries.search_fuzzy(country)
                country_obj = c[0]
            continent_code = pc.country_alpha2_to_continent_code(country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'\
            or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South' or country == 'South Korea':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            else:
                return country, country
            country_obj = pycountry.countries.search_fuzzy(country)
            continent_code = pc.country_alpha2_to_continent_code(country_obj[0].alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj[0].alpha_3, continent
    
    def get_iso3(self, country):
        return self.d[country]['code']
    
    def get_continent(self,country):
        return self.d[country]['continent']
    
    def add_values(self,country):
        self.d[country] = {}
        self.d[country]['code'],self.d[country]['continent'] = self.get_country_details(country)
    
    def fetch_iso3(self,country):
        if country in self.d.keys():
            return self.get_iso3(country)
        else:
            self.add_values(country)
            return self.get_iso3(country)
        
    def fetch_continent(self,country):
        if country in self.d.keys():
            return self.get_continent(country)
        else:
            self.add_values(country)
            return self.get_continent(country)
#added temp variance to weather_country
weather_country = pd.read_csv('/kaggle/input/weather-data-5/training_data_with_weather_info_week_5.csv', parse_dates=['Date'])
weather_country = weather_country.rename(columns={"country+province": "country_province"})
weather_country['Date'] = pd.to_datetime(weather_country['Date'], format = '%Y-%m-%d')
weather_country['temp variance'] = weather_country['max'] - weather_country['min']
weather_country = weather_country.drop(columns = ['stp', 'slp', 'dewp'])
#weather_country['LatLong'] = "("+ str(weather_country['Lat'].round(5)) + "," + str(weather_country['Long'].round(5)) + ")"
#weather_country['LatLong'].sample(10)
weather_country["latlong"] = list(zip(weather_country.Lat.round(6), weather_country.Long.round(6)))
weather_country.sample(1)

#c = weather_country.loc[weather_country['Country_Region'] == 'US']

#added population and density per country to weather_country
pp = pd.read_csv("/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv")
pop = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")
# select only population
pop = pop.iloc[:, :10]
# rename column names
pop.columns = ['Country_Region', 'Population', 'Year Change', 'Net Change', 'Density P/km^2', 'Land Area','Migrants','Fert. Rate','Med. Age', 'Urban Pop %']
pop = pop.drop(columns = ['Year Change', 'Net Change','Land Area','Migrants','Fert. Rate'])
pop['Urban Pop %'] = pop['Urban Pop %'].replace({'N.A.':'0 %'})
pop['Urban Pop %'] = pop['Urban Pop %'].str.rstrip('%').astype('float')

# update populaion
cols = ['Burma', 'Congo (Brazzaville)', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Czechia', 
        'Kosovo', 'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines', 
        'Taiwan*', 'US', 'West Bank and Gaza']
pops = [54409800, 89561403, 5518087, 26378274, 10708981, 1793000, 
        53109, 110854, 23806638, 330541757, 4569000]
dense = [83, 16, 16, 83, 138, 159, 205, 284, 673, 35, 758]
medage = [29.2, 16.7, 19.5, 20.3, 43.3, 30.5, 36.5, 35.3, 42.3, 38.5, 21.9]
urbanpop = [31, 44, 67, 51, 74, 50, 31, 52, 78, 82, 76]
new_df = pd.DataFrame({'Country_Region': cols, 'Population': pops, 'Density P/km^2': dense, 'Med. Age': medage, 'Urban Pop %': urbanpop})
pop.update(new_df)
pop.replace(['South Korea', 'North Korea'], ['Korea, South', 'Korea, North'])
# merged data
weather_country = pd.merge(weather_country, pop, on='Country_Region', how='left')
weather_country['Population'] = weather_country['Population'].fillna(0)
weather_country.sample(3)


#display(weather_country.loc[weather_country['Population'] != 0.0])

#filled the population and density for places that weren't recorded 
for i in weather_country.index:
    if weather_country.loc[i, "Country_Region"] == 'Bangladesh':
        weather_country.loc[i, "Population"] = 164336258
        weather_country.loc[i, "Density P/km^2"] = 1265
        weather_country.loc[i, "Med. Age"] = 27.9
        weather_country.loc[i, "Urban Pop %"] = 37
    elif weather_country.loc[i, "Country_Region"] == 'Brazil':
        weather_country.loc[i, "Population"] = 212228418
        weather_country.loc[i, "Density P/km^2"] = 25
        weather_country.loc[i, "Med. Age"] = 33.2
        weather_country.loc[i, "Urban Pop %"] = 87
    elif weather_country.loc[i, "Country_Region"] == 'Kuwait':
        weather_country.loc[i, "Urban Pop %"] = 100
    elif weather_country.loc[i, "Country_Region"] == 'Holy See':
        weather_country.loc[i, "Urban Pop %"] = 100
        weather_country.loc[i, "Med. Age"] = 60
    elif weather_country.loc[i, "Country_Region"] == 'Andorra':
        weather_country.loc[i, "Med. Age"] = 44.9
    elif weather_country.loc[i, "Country_Region"] == 'San Marino':
        weather_country.loc[i, "Med. Age"] = 44.5
    elif weather_country.loc[i, "Country_Region"] == 'Dominica':
        weather_country.loc[i, "Med. Age"] = 34
    elif weather_country.loc[i, "Country_Region"] == 'Liechtenstein':
        weather_country.loc[i, "Med. Age"] = 43.4        
    elif weather_country.loc[i, "Country_Region"] == 'Diamond Princess':
        weather_country = weather_country.drop(index=i)
    elif weather_country.loc[i, "Country_Region"] == 'India':
        weather_country.loc[i, "Population"] = 1380004385 
        weather_country.loc[i, "Density P/km^2"] = 464
        weather_country.loc[i, "Urban Pop %"] = 34
        weather_country.loc[i, "Med. Age"] = 28.7
    elif weather_country.loc[i, "Country_Region"] == 'China':
        weather_country.loc[i, "Population"] = 1438116346 
        weather_country.loc[i, "Density P/km^2"] = 153
        weather_country.loc[i, "Med. Age"] = 38.4
        weather_country.loc[i, "Urban Pop %"] = 59
    elif weather_country.loc[i, "Country_Region"] == 'West Bank and Gaza':
        weather_country.loc[i, "Population"] = 4569000 
        weather_country.loc[i, "Density P/km^2"] = 758
        weather_country.loc[i, "Med. Age"] = 21.9
        weather_country.loc[i, "Urban Pop %"] = 76
    elif weather_country.loc[i, "Country_Region"] == 'Indonesia':
        weather_country.loc[i, "Population"] = 272884327 
        weather_country.loc[i, "Density P/km^2"] = 151
        weather_country.loc[i, "Med. Age"] = 31.1
        weather_country.loc[i, "Urban Pop %"] = 55
    elif weather_country.loc[i, "Country_Region"] == 'US':
        weather_country.loc[i, "Urban Pop %"] = 82
    elif weather_country.loc[i, "Country_Region"] == 'Venezuela':
        weather_country.loc[i, "Urban Pop %"] = 88
    elif weather_country.loc[i, "Country_Region"] == 'Monaco':
        weather_country.loc[i, "Urban Pop %"] = 100
        weather_country.loc[i, "Med. Age"] = 53.1
    elif weather_country.loc[i, "Country_Region"] == 'Singapore':
        weather_country.loc[i, "Urban Pop %"] = 100
    elif weather_country.loc[i, "Country_Region"] == 'Japan':
        weather_country.loc[i, "Population"] = 126559084 
        weather_country.loc[i, "Density P/km^2"] = 347
        weather_country.loc[i, "Med. Age"] = 48.6
        weather_country.loc[i, "Urban Pop %"] = 92
    elif weather_country.loc[i, "Country_Region"] == 'Sao Tome and Principe':
        weather_country.loc[i, "Population"] = 218241
        weather_country.loc[i, "Density P/km^2"] = 228
        weather_country.loc[i, "Med. Age"] = 19.3
        weather_country.loc[i, "Urban Pop %"] = 73
    elif weather_country.loc[i, "Country_Region"] == 'Korea, South':
        weather_country.loc[i, "Population"] = 51259674 
        weather_country.loc[i, "Density P/km^2"] = 527
        weather_country.loc[i, "Med. Age"] = 43.2
        weather_country.loc[i, "Urban Pop %"] = 81
    elif weather_country.loc[i, "Country_Region"] == 'Russia':
        weather_country.loc[i, "Population"] = 145920988 
        weather_country.loc[i, "Density P/km^2"] = 9
        weather_country.loc[i, "Med. Age"] = 40.3
        weather_country.loc[i, "Urban Pop %"] = 74
    elif weather_country.loc[i, "Country_Region"] == 'MS Zaandam':
        weather_country = weather_country.drop(index=i)
    elif weather_country.loc[i, "Country_Region"] == 'Korea, South':
        weather_country = weather_country.drop(index=i)
    elif weather_country.loc[i, "Country_Region"] == 'Pakistan':
        weather_country.loc[i, "Population"] = 219922471 
        weather_country.loc[i, "Density P/km^2"] = 287
        weather_country.loc[i, "Med. Age"] = 22
        weather_country.loc[i, "Urban Pop %"] = 37
    elif weather_country.loc[i, "Country_Region"] == 'Mexico':
        weather_country.loc[i, "Population"] = 128633396 
        weather_country.loc[i, "Density P/km^2"] = 66
        weather_country.loc[i, "Med. Age"] = 29.3
        weather_country.loc[i, "Urban Pop %"] = 80
    elif weather_country.loc[i, "Country_Region"] == 'Nigeria':
        weather_country.loc[i, "Population"] = 204968096
        weather_country.loc[i, "Density P/km^2"] = 226
        weather_country.loc[i, "Med. Age"] = 18.6
        weather_country.loc[i, "Urban Pop %"] = 50
    else:
        weather_country.loc[i, "Country_Region"] = weather_country.loc[i, "Country_Region"]  

# Cases per population 
weather_country['Urban Pop %'].fillna(0, inplace=True)
weather_country['Urban Pop %'] = weather_country['Urban Pop %']/ 100.0
weather_country['Cases_Million_People'] = round((weather_country['ConfirmedCases'] / weather_country['Population']) * 1000000)
weather_country['ln(Cases / Million People)'] = np.log(weather_country.Cases_Million_People + 1)

weather_country.sample(3)


eastasia  = ['Taiwan*', 'Taiwan', 'Mongolia', 'China', 'Japan', 'South Korea']
europe = ['Latvia', 'Switzerland', 'Liechtenstein', 'Italy', 'Norway', 'Austria', 'Albania',
          'United Kingdom', 'Iceland', 'Finland', 'Luxembourg', 'Belarus', 'Bulgaria', 
          'Guernsey', 'Poland', 'Moldova', 'Spain', 'Bosnia and Herzegovina', 'Portugal', 
          'Germany', 'Monaco', 'San Marino', 'Andorra', 'Slovenia', 'Montenegro', 'Ukraine',
          'Lithuania', 'Netherlands', 'Slovakia', 'Czechia', 'Malta', 'Hungary', 'Jersey', 
          'Serbia', 'Kosovo', 'France', 'Croatia', 'Sweden', 'Estonia', 'Denmark', 
          'North Macedonia', 'Greece', 'Ireland', 'Romania', 'Belgium']
a = []
for i in weather_country.index:
    if weather_country.loc[i, "Country_Region"] in eastasia:
        a.append("East Asia")
    elif weather_country.loc[i, "Country_Region"] in europe:
        a.append("Europe")
    elif weather_country.loc[i, "Country_Region"] == 'US':
        a.append("US")
    else:
        a.append("Rest Of World")

weather_country["Country_Group"] = a
weather_country.sample(1)
second = get2deriv(weather_country, 'Date', 'latlong', 'ConfirmedCases')
weather_country = pd.merge(weather_country, second, on=['latlong','Date'], how='left')

daily = getdaily(weather_country, 'Date', 'latlong', 'ConfirmedCases')
weather_country = pd.merge(weather_country, daily, on=['latlong','Date'], how='left')

weather_country["daily_case"] = weather_country["daily_case"].fillna(0.0)

first = get1deriv(weather_country, 'Date', 'latlong', 'daily_case')
weather_country = pd.merge(weather_country, first, on=['latlong','Date'], how='left')

lag = getlag(weather_country, 'Date', 'latlong', 'ConfirmedCases')
weather_country = pd.merge(weather_country, lag, on=['latlong','Date'], how='left')

lag2 = getlagdeath(weather_country, 'Date', 'latlong', 'Fatalities')
weather_country = pd.merge(weather_country, lag2, on=['latlong','Date'], how='left')

weather_country["lag"] = weather_country["lag"].fillna(0.0)
weather_country["lag_death"] = weather_country["lag_death"].fillna(0.0)

first = get1deriv(weather_country, 'Date', 'latlong', 'lag')
first = first.rename(columns={"1stderiv": "lag_1stderiv"})
weather_country = pd.merge(weather_country, first, on=['latlong','Date'], how='left')


weather_country["2ndderiv"] = weather_country["2ndderiv"].fillna(1.0)
weather_country["1stderiv"] = weather_country["1stderiv"].fillna(0.0)
weather_country["lag"] = weather_country["lag"].fillna(0.0)
weather_country["lag_death"] = weather_country["lag_death"].fillna(0.0)
weather_country["lag_1stderiv"] = weather_country["lag_1stderiv"].fillna(0.0)
#display(weather_country.loc[weather_country['Country_Region'] == 'Germany'])
weather_country.sample(3)
updated = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate'])
latlongupdate = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
updated["countstate"] = tuple(zip(updated['Country/Region'], updated['Province/State']))
updated['Province/State'].replace(np.nan, "Not Reported", inplace=True)
#updated["ObservationDate"]=pd.to_datetime(updated["ObservationDate"])
updated = updated.rename(columns={"ObservationDate": "Date"})
updated['Date'] = pd.to_datetime(updated['Date'], format = '%Y-%m-%d')

a = updated[['Country/Region', 'Province/State']]

a = a.drop_duplicates()
a.reset_index()
a['identify'] = a.index

#adding active cases 
updated = pd.merge(updated, a, on=['Country/Region', 'Province/State'], how='left')
updated['active cases'] = updated['Confirmed'] - updated['Recovered'] - updated['Deaths']

#adding derivatives 
second = get2deriv(updated, 'Date', 'identify', 'Confirmed')
updated = pd.merge(updated, second, on=['identify','Date'], how='left')

first = get1deriv(updated, 'Date', 'identify', 'Confirmed')
updated = pd.merge(updated, first, on=['identify','Date'], how='left')

updated["2ndderiv"] = updated["2ndderiv"].fillna(1.0)
updated["1stderiv"] = updated["1stderiv"].fillna(0.0)

#adding regions
eastasia  = ['Taiwan*', 'Taiwan', 'Mongolia', 'China', 'Mainland China','Japan', 'Korea, South','South Korea', 'Hong Kong']
europe = ['Latvia', 'Switzerland', 'Liechtenstein', 'Italy', 'Norway', 'Austria', 'Albania',
          'United Kingdom', 'Iceland', 'Finland', 'Luxembourg', 'Belarus', 'Bulgaria', 
          'Guernsey', 'Poland', 'Moldova', 'Spain', 'Bosnia and Herzegovina', 'Portugal', 
          'Germany', 'Monaco', 'San Marino', 'Andorra', 'Slovenia', 'Montenegro', 'Ukraine',
          'Lithuania', 'Netherlands', 'Slovakia', 'Czechia', 'Malta', 'Hungary', 'Jersey', 
          'Serbia', 'Kosovo', 'France', 'Croatia', 'Sweden', 'Estonia', 'Denmark', 
          'North Macedonia', 'Greece', 'Ireland', 'Romania', 'Belgium', 'UK']
a = []
updated['country_group'] = updated['Confirmed']
updated['country_group'] = 0

for i in updated.index:
    if updated.loc[i, "Country/Region"] in eastasia:
        updated.loc[i, "country_group"] = "East Asia"
    elif updated.loc[i, "Country/Region"] in europe:
        updated.loc[i, "country_group"] = "Europe"
    elif updated.loc[i, "Country/Region"] == 'US':
        updated.loc[i, "country_group"] = "US"
    else:
        updated.loc[i, "country_group"] = "Rest Of The World"

updated[updated['country_group']=="Rest Of The World"].head(3)


updated.sample(3)
"""
#latlongupdate['countstate'] = tuple(zip(latlongupdate['Country/Region'], latlongupdate['Province/State']))
just_latlong = latlongupdate[['Country/Region','Province/State', 'Lat', 'Long']].copy()

#display(updated[updated['Country/Region']=="South Korea"])
just_latlong = just_latlong.replace('Korea, South', 'South Korea')


#display(just_latlong[just_latlong['Country/Region'].str.contains("Korea")])
just_latlong['Province/State'].replace(np.nan, "Not Reported", inplace=True)

wc = weather_country[['Country_Region', 'Province_State', 'Lat', 'Long']].copy()
wc = wc.rename(columns={'Country_Region': "Country/Region",'Province_State': 'Province/State' })
wc['Province/State'].replace(np.nan, "Not Reported", inplace=True)

nota = wc[~wc['Province/State'].isin(just_latlong['Province/State'])]
nota = nota.groupby(['Country/Region','Province/State'],as_index=False)['Lat','Long'].mean()
c = ['US', 'Macau','UK', 'UK', 'US', 'UK', 'UK', 'UK', 'UK', 'Germany', 'US', 'US', 'UK', 'UK', 'UK', 'UK', 'UK',
     'Czech Republic', 'The Bahamas', 'Republic of the Congo', 'Ivory Coast', 'Netherlands', 'Denmark', 'Palestine']
r =['Chicago', 'Macau', 'Isle of Man', 'Montserrat', 'Northern Mariana Islands', 'Turks and Caicos Islands', 
    'Falkland Islands (Malvinas)', 'Gibraltar','Cayman Islands', 'Bavaria', 'American Samoa', 'United States Virgin Islands', 
    'Channel Islands', 'Bermuda', 'Anguilla', 'British Virgin Islands', 'Not Reported', 'Not Reported', 'Not Reported', 'Not Reported', 'Not Reported',
   'Netherlands', 'Denmark', 'Not Reported']
la=[41.8339042,23.6356074, 54.2278829, 16.691357, 17.3076967, 21.5741504, -51.7206292, 36.1295735, 19.5081819, 48.8992765, -14.061727, 18.0672779,
   49.4582161, 32.3194245, 18.390315,18.5222738, 52.7602022, 49.7856662, 24.4229244, -0.6811523, 7.4662967, 52.1951016, 56.2128538, 31.8858324]
lo=[-88.0121503,114.4376334,-4.8523185,-60.2272795,143.2420346,-72.3505781,-60.6489884,-5.3883195,-81.1347306,9.1651538,-170.6672906,-65.2991668,
    -2.942905,-64.8364403,-63.4803453,-64.7114365,-6.813662,13.2321306,-78.2108881,10.3858451,-7.7921532,3.0367463,9.3001434,34.331614]
pp = pd.DataFrame(data={'Country/Region': c, 'Province/State':r, 'Lat':la, 'Long':lo})
just_latlong = just_latlong.append(nota)
just_latlong = just_latlong.append(pp)

just_latlong.sample(10)
print(len(c))
print(len(r))
print(len(la))
print(len(lo))

#with_updated.sample(2)
with_updated.isnull().any()
with_updated['bool_loc'] = pd.notnull(with_updated["Lat"]) 
a = with_updated[with_updated['Lat']=='Not Reported']
#a[a['Country/Region']!='UK'].sample(50)
#a = with_updated[with_updated['Country/Region']=='UK']
#a[a['Province/State']=='Not Reported'].sample(50)
#b[b['Province/State']== 'United Kingdom'].sample(7)
#display(with_updated[with_updated['Country/Region']=='Czechia'])
print(len(a))
print(len(with_updated))

"""
local = weather_country.copy()
local.replace(np.inf, np.nan).fillna(0)

local = local.groupby(['Country_Region','Province_State','latlong', 'Lat', 'Long'],as_index=False)['1stderiv','2ndderiv'].mean()
local['sderiv'] = local['2ndderiv']
p = local[local.sderiv >= 1.2]
f = local[local.sderiv <= 0.8]
t = local['sderiv'] <1.2
a = local['sderiv'] > 0.8
g = local[t & a]

p['sderiv'] = p['2ndderiv']
p['ln(2ndderiv)'] = np.log(p.sderiv + 1)
px.set_mapbox_access_token('pk.eyJ1IjoibW9ydXRraW4iLCJhIjoiY2s1cnJhMzczMGdjaDNtcnR0M2h0NnR6cSJ9.87UtlJlwluWbZq4ioist-g')
df = px.data.carshare()
p['sderiv'] = p['2ndderiv']*10
fig = px.scatter_mapbox(p, lat="Lat", lon="Long",     color="2ndderiv", size="2ndderiv",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=50, zoom=3)
fig.update_layout(
    hovermode='closest',
    title_text= "Locations With An Accelerating Growth Rate (GR > 1.2) on 03/07-03/13",
    mapbox=dict(
        center=go.layout.mapbox.Center(
            lat=41.1254,
            lon=-98.2651
        ),
        pitch=0,
        zoom=3
    )
)
fig.update_layout(legend_title_text='Growth Rate')

fig.show()
world = updated.copy()
world = updated.groupby(['Date'],as_index=False)['active cases', 'Confirmed', 'Deaths'].sum()
world['active cases'][world['active cases'] < 0] = 0



fig = go.Figure(data=[
    go.Bar(name='Cases', x=world['Date'], y=world['active cases']),
    go.Bar(name='Deaths', x=world['Date'], y=world['Deaths'])
])
# Change the bar mode
fig.update_layout(
    title='Daily Cases Growth',
    xaxis_title="Date",
    yaxis_title="# of Daily Cases and Deaths",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig.update_layout(barmode='overlay',
                  title = {'y':0.9,
        'x':0.5,'text':'Worldwide Daily Cases and Deaths count over time', 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=world['Date'], y=world['active cases'],
                    mode='lines',
                    name='active cases'))


fig1.update_layout(
    title='Daily Cases Growth',
    xaxis_title="Date",
    yaxis_title="# of Daily Cases",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig1.update_layout(
    title={
        'text': 'World Daily Case Count over time',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)

fig1.show()


cases1 = updated.copy()
#cases['rh'] = cases['rh'].fillna(0)
grp4 = cases1.groupby(['Date', 'Country/Region'])['active cases', 'Confirmed', 'Deaths', 'Recovered'].sum()

grp4['ln_ConfirmedCases'] = np.log(grp4.Confirmed + 1) 
grp4 = grp4.reset_index()
grp4['Date'] = grp4['Date'].dt.strftime('%m/%d/%Y')
grp4['Country'] =  grp4['Country/Region']

fig = px.scatter_geo(grp4, locations="Country", locationmode='country names', 
                     color="ln_ConfirmedCases",size= "ln_ConfirmedCases", hover_name="Country/Region",hover_data = [grp4.Confirmed, grp4.Deaths, grp4.Recovered ],projection="natural earth",
                     animation_frame="Date",width=900, height=700,
                        color_continuous_scale="portland",
                     title='World Map of Log Coronavirus Cases Change Over Time (14 = 1.18M)')

fig.update(layout_coloraxis_showscale=True)

cases = weather_country.copy()
cases['rh'] = cases['rh'].fillna(0)
grp2 = cases.groupby(['Date', 'Country_Region'])['ConfirmedCases', 'daily_case', 'Fatalities'].sum()
grp1 = cases.groupby(['Date', 'Country_Region'])['rh', 'temp'].mean()
grp = pd.merge(grp2, grp1, on=['Country_Region', 'Date'], how='left')

grp['ln_ConfirmedCases'] = np.log(grp.ConfirmedCases + 1) 
grp = grp.reset_index()
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Country'] =  grp['Country_Region']


grp['dumtemp'] = grp['temp']+50
grp['dumtemp'] = grp['dumtemp'].round(2)

fig = px.scatter_geo(grp, locations="Country", locationmode='country names', 
                     color="temp",size= "dumtemp", hover_name="Country_Region",hover_data = [grp.ConfirmedCases, grp.daily_case, grp.Fatalities ],projection="natural earth",
                     animation_frame="Date",width=900, height=700,
                    color_continuous_scale="portland", title='COVID-19: Temperature By Country/Region Over Time')

fig.update(layout_coloraxis_showscale=True)
fig.show()

fig1 = px.scatter_geo(grp, locations="Country", locationmode='country names', 
                     color="rh",size= "rh", hover_name="Country_Region",hover_data = [grp.ConfirmedCases, grp.daily_case, grp.Fatalities ],projection="natural earth",
                     animation_frame="Date",width=900, height=700,
                        color_continuous_scale="portland",
                     title='COVID-19: Relative Humidity By Country/Region Over Time')

fig1.update(layout_coloraxis_showscale=True)
fig1.show()
df = weather_country.copy()
df['latlongcount'] = list(zip(df.Country_Region, df.Province_State))
#ind = df['latlongcount'].to_numpy()
new = df[['Date','Country_Region','latlongcount', '1stderiv', 'Density P/km^2', 'temp', 'ConfirmedCases', 'Population']].copy()
#new['Date'] = pd.tslib.Timestamp(new['Date'])
new = new[new.Date == new.Date.max()]

new1 = new.groupby(['Country_Region'],as_index=False)['1stderiv', 'Population', 'Density P/km^2', 'temp'].mean()
new2 = new.groupby(['Country_Region'],as_index=False)['ConfirmedCases'].sum()
new = pd.merge(new1, new2, on='Country_Region', how='left')
new['Cases_Million_People'] = round((new['ConfirmedCases'] / new['Population']) * 1000000)

#df = px.data.gapminder()
fig = px.scatter(new, x="Density P/km^2", y="1stderiv",
           size="Cases_Million_People", color="temp", hover_name="Country_Region",
           log_y=True, log_x=True, size_max=55)
fig.update_layout(title='Rate of New Cases / Density (Pop/km^2) With Temperature(C) Color Gradient')
fig.update_layout(
    title={
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)
c = weather_country.copy()
c['Province_State'] = c['Province_State'].fillna("Total")
#c = c.drop(columns = ['1stderiv', '2ndderiv'])
c2 = c.groupby(['Date', 'Country_Group', 'Country_Region'],as_index=False)['daily_case'].sum()
c = c.groupby(['Date', 'Country_Group', 'Country_Region'],as_index=False)['temp'].mean()
c = pd.merge(c, c2, on=['Date', 'Country_Group', 'Country_Region'], how='left')
c = c[['Date','Country_Group','Country_Region', 'daily_case', 'temp']]

europe = c[c['Country_Group']=="Europe"]
europe_n = europe[europe['Country_Region'].isin(['France', 'Germany', 'Netherlands', 'United Kingdom', 'Spain', 'Belgium'])]

fig5 = px.line(europe_n, x="Date", y="temp", color='Country_Region', title='Average Temperature (C) Change Over Time Per Country In Europe')
fig5.update_layout(
    xaxis_title="Date",
    yaxis_title="Temperature (C)",
    title={
        'y':0.9,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'}
    #font=dict(
    #    family="Courier New, monospace",
    #    size=14,
    #    color="#7f7f7f"
    #)
)
fig5.show()

fig6 = px.line(europe_n, x="Date", y="daily_case", color='Country_Region', title='Average Rate Of Cases Change Over Time In Countries In Europe')
fig6.update_layout(
    xaxis_title="Date",
    yaxis_title="Rate of Daily Cases",
    title={
        'y':0.9,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'}
    #font=dict(
    #    family="Courier New, monospace",
    #    size=14,
    #    color="#7f7f7f"
    #)
)
fig6.show()
updated_groups = updated.groupby(['Date', 'country_group'],as_index=False)['active cases', 'Confirmed'].sum()
updated_groups['active cases'][updated_groups['active cases'] < 0] = 0
updated_groups = updated_groups.groupby('country_group').resample('W-Mon', on='Date').sum().reset_index().sort_values(by='Date')
updated_groupsl = updated_groups.copy()

wd1 = weather_country.groupby(['Date', 'Country_Group'],as_index=False)['daily_case', 'ConfirmedCases'].sum()
wd1['daily_case'][wd1['daily_case'] < 0] = 0
wd1 = wd1.groupby('Country_Group').resample('W-Mon', on='Date').sum().reset_index().sort_values(by='Date')
wd2 = wd1.copy()

#added mean temp 
a = weather_country.copy()
updated_groups3 = a.groupby(['Date', 'Country_Group'],as_index=False)['temp'].mean()
updated_groups3 = updated_groups3.rename(columns={"Country_Group": "country_group"})
updated_groups = pd.merge(updated_groups, updated_groups3, on=['country_group','Date'], how='left')
updated_groups['temp'] = updated_groups['temp'].fillna("hi")

#only kept the data points where temp is recorded 
updated_groups = updated_groups[updated_groups.temp != 'hi']

#adding derivatives for the regions UPDATEDGROUPS
second = get2deriv(updated_groupsl, 'Date', 'country_group', "active cases")
updated_groupsl = pd.merge(updated_groupsl, second, on=['country_group','Date'], how='left')

first = get1deriv(updated_groupsl, 'Date', 'country_group', "active cases")
updated_groupsl = pd.merge(updated_groupsl, first, on=['country_group','Date'], how='left')

updated_groupsl["2ndderiv"] = updated_groupsl["2ndderiv"].fillna(1.0)
updated_groupsl["1stderiv"] = updated_groupsl["1stderiv"].fillna(0.0)

#adding derivatives for the regions WEATHER-COUNTRY
second = get2deriv(wd2, 'Date', 'Country_Group', "daily_case")
wd2 = pd.merge(wd2, second, on=['Country_Group','Date'], how='left')

first = get1deriv(wd2, 'Date', 'Country_Group', "daily_case")
wd2 = pd.merge(wd2, first, on=['Country_Group','Date'], how='left')

wd2["2ndderiv"] = wd2["2ndderiv"].fillna(1.0)
wd2["1stderiv"] = wd2["1stderiv"].fillna(0.0)

updated_groupsl = updated_groupsl.loc[updated_groupsl['Date'] <='2020-05-11']



##########using weather data ###############
wupdated = weather_country.copy()
wupdated_groups1 = wupdated.groupby(['Date', 'Country_Group'],as_index=False)['daily_case', 'ConfirmedCases'].sum()
wupdated_groups2 = wupdated.groupby(['Date', 'Country_Group'],as_index=False)['rh', 'temp'].mean()
wupdated_groups = pd.merge(wupdated_groups1, wupdated_groups2, on=['Country_Group','Date'], how='left')
wupdated_groups = wupdated_groups.groupby('Country_Group').resample('W-Mon', on='Date').sum().reset_index().sort_values(by='Date')

second = get2deriv(wupdated_groups, 'Date', 'Country_Group', 'ConfirmedCases')
wupdated_groups = pd.merge(wupdated_groups, second, on=['Date', 'Country_Group'], how='left')

first = get1deriv(wupdated_groups, 'Date', 'Country_Group', 'daily_case')
wupdated_groups = pd.merge(wupdated_groups, first, on=['Date', 'Country_Group'], how='left')

wupdated_groups["2ndderiv"] = wupdated_groups["2ndderiv"].fillna(1.0)
wupdated_groups["1stderiv"] = wupdated_groups["1stderiv"].fillna(0.0)
fig = px.line(updated_groupsl, x="Date", y="active cases", color='country_group')



fig.update_layout(
    title='Regional Daily Cases over time until 05/11',
    xaxis_title="Date",
    yaxis_title="# of Daily Cases",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig.show()
fig = px.line(updated_groupsl, x="Date", y="1stderiv", color='country_group')



fig.update_layout(
    title='Regional Daily Rate Of Change over time until 5/11',
    xaxis_title="Date",
    yaxis_title="# of Daily Cases Rate",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig.show()
ca = weather_country.copy()
ca['Province_State'] = ca['Province_State'].fillna("Total")
#c = c.drop(columns = ['1stderiv', '2ndderiv'])
cb = ca.groupby(['Date', 'Country_Group'],as_index=False)['daily_case'].sum()
ca = ca.groupby(['Date', 'Country_Group'],as_index=False)['temp'].mean()
ca = pd.merge(ca, cb, on=['Date', 'Country_Group'], how='left')

fig1 = px.line(ca, x="Date", y="temp", color='Country_Group', title='Average Temperature Change Over Time Per Region until 4/11')
fig1.update_layout(
    xaxis_title="Date",
    yaxis_title="Temperature (C)",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig1.show()

#fig2 = px.line(ca, x="Date", y="daily_case", color='Country_Group', title='Average Temperature Change Over Time Per Region')
#fig2.show()

df = weather_country.copy()
df['latlongcount'] = df['Country_Region'] + ',' + df['Province_State']
#ind = df['latlongcount'].to_numpy()
new = df[['Date','latlongcount', 'Country_Group','lag_1stderiv', 'lag','Density P/km^2', 'temp', 'ConfirmedCases', 'Population', 'rh']].copy()
#new['Date'] = pd.tslib.Timestamp(new['Date'])
new = new[new.Date > '2020-04-30']
new = new[new.Date < '2020-05-05']

new1 = new.groupby(['latlongcount', 'Country_Group'],as_index=False)['lag_1stderiv', 'lag','Population', 'Density P/km^2', 'temp', 'rh'].mean()
new2 = new.groupby(['latlongcount'],as_index=False)['ConfirmedCases'].max()
new = pd.merge(new1, new2, on='latlongcount', how='left')
new['Cases_Million_People'] = round((new['ConfirmedCases'] / new['Population']) * 1000000)

#df = px.data.gapminder()
fig = px.scatter(new, x="temp", y="lag_1stderiv",
            color="Country_Group", hover_name="latlongcount",
           log_y=True,log_x=True, size_max=55)
fig.update_layout(title='4/30-5/05 Average Rate of New Cases / Temperature(C)')
fig.update_layout(
    xaxis_title="Temperature (C)",
    yaxis_title="Case Rate",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig.show()
fig1 = px.scatter(new, x="rh", y="lag_1stderiv",
            color="Country_Group", hover_name="latlongcount",
           log_y=True,log_x=True, size_max=55)
fig1.update_layout(title='4/30-5/05 Average Rate of New Cases / Realtive Humidity')
fig1.update_layout(
    xaxis_title="Relative Humidity",
    yaxis_title="Case Rate",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig1.show()
fig = px.scatter_3d(new, x='rh', y='temp', z='ConfirmedCases',
              color='Country_Group', log_z=True)
fig.update_layout(title='4/3-4/11 Average Rate of New Cases / Temp (C) / Relative Humidity')
fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig.show()
weather_country.tail(3)
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def lin_reg(X_train, Y_train, X_test, regr):
    # Create linear regression object
    #regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    param = regr.get_params()

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    
    return regr, y_pred

data = weather_country.copy()
# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends
data['ConfirmedCases'] = data['ConfirmedCases'].astype('float64')
data['Fatalities'] = data['Fatalities'].astype('float64')

# Replace infinites
data.replace([np.inf, -np.inf], 0, inplace=True)
data = data.fillna(0)
features = ['Country_Region','Province_State','ConfirmedCases','Population','wdsp', 'prcp','fog','temp', 'rh', 'min', 'max','temp variance','Density P/km^2',
       'day_from_jan_first','Med. Age','Urban Pop %' ]
data = data[features]

data.sort_values(by=['day_from_jan_first'])
train = data.loc[data['day_from_jan_first'] <=109]
test = data.loc[data['day_from_jan_first'] > 110]

trainy = train[['ConfirmedCases']].copy()
trainy_nolog = trainy.copy()
trainy = trainy.apply(lambda x: np.log1p(x))

trainx = train.drop(columns = ['Country_Region', 'Province_State','ConfirmedCases']).copy()

testy = test[['ConfirmedCases']].copy()
testy_nolog = test[['ConfirmedCases']].copy()
testy = testy.apply(lambda x: np.log1p(x))

testx = test.drop(columns = ['Country_Region', 'Province_State','ConfirmedCases']).copy()



data.sample(10)
param = 0

regr = linear_model.LinearRegression()
r, ypred= lin_reg(trainx, trainy, testx, regr)

r2 = r2_score(testy, ypred)
mse = mean_absolute_error(testy, ypred)
#msle = mean_squared_log_error(testy, ypred)
print("R squared Value (Variability Explained)")
print(r2)
print(' ')
print("Mean Absolute Error Value (Variance)")
print(mse)



s = test.copy()
s['predicted'] = ypred
s['real'] = testy
a = s.groupby(['day_from_jan_first'],as_index=False)['predicted', 'real'].sum()
#s.head(10)

fig = go.Figure()
fig.add_trace(go.Scatter(x=a['day_from_jan_first'], y=a['real'],
                    mode='lines',
                    name='Real Confirmed Cases'))
fig.add_trace(go.Scatter(x=a['day_from_jan_first'], y=a['predicted'],
                    mode='lines',
                    name='Predicted Confirmed Cases' ))

#fig.add_shape(
#        # Line Horizontal
#            type="line",
#            x0=90,
#            y0=1600,
#            x1=102,
#            y1=1600,
#            line=dict(
#                color="LightSeaGreen",
#                width=2,
#                dash="dashdot",
#            ),
#    )


fig.update_layout(title='Regression Predictive Model: 04/20/2020 - 05/15/2020')
fig.update_layout(
    #font=dict(
    #    family="Courier New, monospace",
    #    size=14,
    #    color="#7f7f7f"
    #)
)


fig.show()

rfcla = RandomForestRegressor(n_estimators=100, max_samples=0.8,
                        random_state=1)
# We train model
rfcla.fit(trainx, trainy)
predictions = rfcla.predict(testx)
#roc_value = roc_auc_score(testy, predictions)

fi = pd.DataFrame({'feature': list(trainx.columns),
                   'importance': rfcla.feature_importances_}).\
                    sort_values('importance', ascending = False)

r2 = r2_score(testy, predictions)
print("R squared Value (Variability Explained)")
print(r2)
print(' ')
print("Mean Squared Error Value (Variance):")
mse = mean_squared_error(testy, predictions)
print(mse)
fi.head()
data1 = weather_country.copy()
# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends
data1['lag']  = data1['lag'].astype('float64')
data1['lag_death'] = data1['lag_death'].astype('float64')
data1['lag'] = data1['lag'].apply(lambda x: np.log1p(x))
data1['lag_death'] = data1['lag_death'].apply(lambda x: np.log1p(x))

# Replace infinites
data1.replace([np.inf, -np.inf], 0, inplace=True)
data1 = data1.fillna(0)
features1 = ['Country_Region','Province_State','lag','Population','wdsp','prcp','fog','temp', 'rh', 'min', 'max','temp variance','Density P/km^2',
       'day_from_jan_first','Med. Age','Urban Pop %' ]
data1 = data1[features1]


#data1.sort_values(by=['day_from_jan_first'])
train1 = data1[data1['day_from_jan_first'] <=104]
test1 = data1[data1['day_from_jan_first'] > 105]
test1 = test1[test1['day_from_jan_first'] < 120]
trainy1 = train1[['lag']].copy()
trainx1 = train1.drop(columns = ['Country_Region', 'Province_State','lag']).copy()
testy1 = test1[['lag']].copy()
testx1 = test1.drop(columns = ['Country_Region', 'Province_State','lag']).copy()
data1.sample(10)
param = 0
regr = linear_model.LinearRegression()
# Linear regression model
def lin_reg(X_train, Y_train, X_test):
    # Create linear regression object
    #regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    param = regr.get_params()

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    
    return regr, y_pred
r1, ypred1= lin_reg(trainx1, trainy1, testx1)

r21 = r2_score(testy1, ypred1)
mse1 = mean_absolute_error(testy1, ypred1)
#msle = mean_squared_log_error(testy, ypred)
print("R squared Value")
print(r21)
print("     ")
print("Mean Absolute Error Value")
print(mse1)
#print("Mean Squared Log Error Value")
#print(msle)
k = test1.copy()


k['predicted'] = ypred1
k['real'] = testy1
d = k.groupby(['day_from_jan_first'],as_index=False)['predicted', 'real'].sum()
#s.head(10)

fig = go.Figure()
fig.add_trace(go.Scatter(x=d['day_from_jan_first'], y=d['real'],
                    mode='lines',
                    name='Real Confirmed Cases'))
fig.add_trace(go.Scatter(x=d['day_from_jan_first'], y=d['predicted'],
                    mode='lines',
                    name='Predicted Confirmed Cases' ))

fig.update_layout(title='Regression Predictive Model WITH 7 day Lag')


fig.show()
rfcla1 = RandomForestRegressor(n_estimators=100, max_samples=0.8,
                        random_state=1)
# We train model
rfcla1.fit(trainx1, trainy1)
predictions1 = rfcla1.predict(testx1)
#roc_value = roc_auc_score(testy, predictions)

fi1 = pd.DataFrame({'feature': list(trainx1.columns),
                   'importance': rfcla1.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi1.head()

r2 = r2_score(testy1, predictions1)
print("R squared Value (Variability Explained)")
print(r2)
print(' ')
print("Mean Squared Error Value (Variance):")
mse = mean_squared_error(testy1, predictions1)
print(mse)
fi.head()
k['rf'] = predictions1
c = k.groupby(['day_from_jan_first'],as_index=False)['real', 'rf'].sum()
fig = go.Figure()
fig.add_trace(go.Scatter(x=c['day_from_jan_first'], y=c['real'],
                    mode='lines',
                    name='Real Confirmed Cases'))
fig.add_trace(go.Scatter(x=c['day_from_jan_first'], y=c['rf'],
                    mode='lines',
                    name='Predicted Confirmed Cases' ))
fig.update_layout(title='Random Forest Predictive Model WITH 7 day Lag')




fig.show()

fig = px.scatter_3d(new, x='rh', y='temp', z='lag',
              color='Country_Group', log_z=True)
fig.update_layout(title='4/3-4/11 Average Rate of New Cases With Lag/ Temp (C) / Relative Humidity')
fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    ),
    scene = dict(
                    xaxis_title='Rel. Humidity',
                    yaxis_title='Temp. (C)',
                    zaxis_title='Cases Rate w/ Lag')
)

fig.show()