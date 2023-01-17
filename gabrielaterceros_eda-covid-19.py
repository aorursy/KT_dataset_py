import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from matplotlib import ticker 
import pycountry_convert as pc
import folium
import branca
from datetime import datetime, timedelta,date
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import json, requests
import calmap
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as tick

import plotly.graph_objs as go
import plotly.offline as py

from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from keras.optimizers import RMSprop, Adam

import warnings
warnings.filterwarnings('ignore')
# Retriving Dataset
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

# Depricated
df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
#Lets start analizing the shape and format of the data
print(df_confirmed.tail(10))
df_confirmed.rename(columns={'Province/State':'state','Country/Region':'country'}, inplace = True)
df_deaths.rename(columns={'Province/State':'state','Country/Region':'country'}, inplace = True)
df_covid19.rename(columns={'Country_Region':'country'}, inplace = True)
# Changing the conuntry names as required by pycountry_convert Lib
df_confirmed.loc[df_confirmed['country'] == "US", "country"] = "USA"
df_deaths.loc[df_deaths['country'] == "US", "country"] = "USA"
df_covid19.loc[df_covid19['country'] == "US", "country"] = "USA"

df_confirmed.loc[df_confirmed['country'] == 'Korea, South', "country"] = 'South Korea'
df_deaths.loc[df_deaths['country'] == 'Korea, South', "country"] = 'South Korea'
df_covid19.loc[df_covid19['country'] == "Korea, South", "country"] = "South Korea"

df_confirmed.loc[df_confirmed['country'] == 'Taiwan*', "country"] = 'Taiwan'
df_deaths.loc[df_deaths['country'] == 'Taiwan*', "country"] = 'Taiwan'
df_covid19.loc[df_covid19['country'] == "Taiwan*", "country"] = "Taiwan"

df_confirmed.loc[df_confirmed['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'
df_deaths.loc[df_deaths['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'
df_covid19.loc[df_covid19['country'] == "Congo (Kinshasa)", "country"] = "Democratic Republic of the Congo"

df_confirmed.loc[df_confirmed['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_deaths.loc[df_deaths['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_covid19.loc[df_covid19['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_confirmed.loc[df_confirmed['country'] == "Reunion", "country"] = "Réunion"
df_deaths.loc[df_deaths['country'] == "Reunion", "country"] = "Réunion"
df_covid19.loc[df_covid19['country'] == "Reunion", "country"] = "Réunion"

df_confirmed.loc[df_confirmed['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'
df_deaths.loc[df_deaths['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'
df_covid19.loc[df_covid19['country'] == "Congo (Brazzaville)", "country"] = "Republic of the Congo"

df_confirmed.loc[df_confirmed['country'] == 'Bahamas, The', "country"] = 'Bahamas'
df_deaths.loc[df_deaths['country'] == 'Bahamas, The', "country"] = 'Bahamas'
df_covid19.loc[df_covid19['country'] == "Bahamas, The", "country"] = "Bahamas"

df_confirmed.loc[df_confirmed['country'] == 'Gambia, The', "country"] = 'Gambia'
df_deaths.loc[df_deaths['country'] == 'Gambia, The', "country"] = 'Gambia'
df_covid19.loc[df_covid19['country'] == "Gambia, The", "country"] = "Gambia"

# getting all countries
countries = np.asarray(df_confirmed["country"])
# Continent_code to Continent_names
continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU' : 'Europe',
    'na' : 'Others'
}

# Defininng Function for getting continent code for country.
def country_to_continent_code(country):
    try:
        a = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
        return continents[a]
    except :
        return 'Others'
    
df_confirmed.insert(1,'continent',df_confirmed.country.map(country_to_continent_code))
df_covid19.insert(1,'continent',df_covid19.country.map(country_to_continent_code))
df_deaths.insert(1,'continent',df_deaths.country.map(country_to_continent_code))
df_confirmed = df_confirmed[df_confirmed.continent != "Others"]
df_covid19 = df_covid19[df_covid19.continent != "Others"]
df_deaths = df_deaths[df_deaths.continent != "Others"]

print(df_covid19.head())
df_countries_cases = df_covid19.copy().drop(['Lat','Long_','Last_Update','ISO3', 'UID'],axis =1)
df_countries_cases.fillna(0,inplace = True)
#No mising values
print(df_countries_cases.isnull().sum())
df_continent = pd.DataFrame(df_covid19.copy().drop(['Lat','Long_','country','Last_Update', 'Incident_Rate', 'UID'],axis =1))
df_continent = df_continent.groupby("continent", as_index = False).sum()
df_continent = df_continent.sort_values("Deaths", ascending = False)

totals = [i+j+k for i,j,k in zip(df_continent.Deaths, df_continent.Recovered, df_continent.Active)]
deathsBars = [i / j * 100 for i,j in zip(df_continent.Deaths, totals)]
recovBars = [i / j * 100 for i,j in zip(df_continent.Recovered, totals)]
activeBars = [i / j * 100 for i,j in zip(df_continent.Active, totals)]
continent = df_continent.continent

df =pd.DataFrame(zip(continent,activeBars, recovBars,deathsBars))
df.sort_values(1, ascending = True, inplace = True)
deathsBars = df[1]
recovBars = df[2]
activeBars = df[3]
y = df[0]

#plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

#my_dpi=90
f = plt.figure(figsize=(7,8))
ax = plt.subplot(211)

width = 0.5
ax.barh(y, deathsBars,width, color=palette(1), edgecolor='white', label='Deaths')
ax.barh(y, recovBars,width, left=deathsBars, color='grey', edgecolor='white', label='Recovers')
ax.barh(y, activeBars,width, left=[100-i for i in activeBars], color=palette(8), edgecolor='white', label='Actives')

ax = plt.gca()
fmt = '%.0f%%' # Format the ticks in %'
xticks = tick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)

plt.title('% of affected by continent (Deaths - Recovered - Actives)', loc='left', fontsize = 18, pad = 10)
ax.set( ylim=[3*width - 1, len(df)+1])    
plt.tick_params(labelsize = 12)   

ax.legend()


plt.show() 
df_dates_cv = df_confirmed.drop(['state','Lat','Long'], axis=1)
dates_columns = df_confirmed.drop(['state','country','continent','Lat','Long'], axis=1)

melted_confirmed=pd.melt(frame=df_dates_cv,id_vars=['country','continent'], value_vars= dates_columns)
melted_confirmed.rename(columns={'variable':'date', 'value':'confirmed'}, inplace= True)
melted_confirmed['deaths'] =0
melted_confirmed
df_dates_cv = df_deaths.drop(['state','Lat','Long'], axis=1)
dates_columns = df_deaths.drop(['state','country','continent','Lat','Long'], axis=1)

melted_deaths = pd.melt(frame=df_dates_cv,id_vars=['country','continent'], value_vars= dates_columns)
melted_deaths.rename(columns={'variable':'date', 'value':'deaths'}, inplace= True)
melted_deaths.insert(3, 'confirmed',0)
melted_deaths
conc_data_row = pd.concat([melted_confirmed, melted_deaths],ignore_index =True)
conc_data_row['date'] = pd.to_datetime(conc_data_row.date)
conc_data_row.insert(3,'month', pd.DatetimeIndex(conc_data_row.date).month)
conc_data_row= conc_data_row.groupby(['date','month'], as_index=False)['confirmed', 'deaths'].sum()

df_month = conc_data_row.groupby(['month','date'], as_index=False).sum()
def y_fmt(tick_val, pos):
    if tick_val > 1000000:
        val = int(tick_val/1000000)
        return '{:d} M'.format(val)
    elif tick_val > 1000:
        val = int(tick_val/ 1000)
        return '{:d} k'.format(val)
    else:
        return tick_val
# Make a data frame
df=pd.DataFrame({'x': df_month.date, 'y1': df_month.confirmed, 'y2': df_month.deaths})
    
#plt.style.use('fivethirtyeight')
plt.style.use('seaborn-darkgrid')
my_dpi=100
#f =plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
f =plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)

palette = plt.get_cmap('Set1')

font = {'family': 'arial',
        'color':  'darkred',
        'weight': 'normal'
        }

# multiple line plot
plt.plot(df['x'], df['y2'], marker='', color='grey', linewidth=1, alpha=0.4)
    
#Now re do the interesting curve, but biger with distinct color
plt.plot(df['x'], df['y1'], marker='', color=palette(1), linewidth=4, alpha=0.7) 

# plot
x_pos = df.x.tail(1).item()+ timedelta(days=5)
plt.text(x_pos, df.y2.tail(1).item(), str(y_fmt(df.y2.tail(1).item(),0)) + ' Deaths', horizontalalignment='left', size=12, color='grey', fontdict=font)
     
#And add a special annotation for the group we are interested in
plt.text(x_pos, df.y1.tail(1).item(), str(df.y1.tail(1).item()/1000) + 'K Confirmed', horizontalalignment='left', size=12, color=palette(1), fontdict=font)

monthyearFmt = mdates.DateFormatter('%d %b')
ax.xaxis.set_major_formatter(monthyearFmt)

ax = plt.gca()
ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
plt.tick_params(labelsize = 13) 

# Add titles
plt.title("Evolution of confirmed cases vs deaths over the time", loc='left', fontsize=18, fontweight=0, color='black', fontdict=font)


conc_data_row = pd.concat([melted_confirmed, melted_deaths],ignore_index =True)
#conc_data_row = conc_data_row[conc_data_row['country'] == 'Spain']
conc_data_row = conc_data_row.groupby(['country','continent','date'], as_index=False)['confirmed', 'deaths'].sum()
conc_data_row['date'] = pd.to_datetime(conc_data_row.date)
#conc_data_row.insert(3,'month', pd.DatetimeIndex(conc_data_row.date).month)

day_after = conc_data_row.copy()
day_after['date'] =  day_after['date']- timedelta(days=1)


conc_data_row_new = pd.merge(conc_data_row,day_after,on=['country','continent','date'])

conc_data_row_new['new_cases'] = conc_data_row_new['confirmed_y'] - conc_data_row_new['confirmed_x']
conc_data_row_new['new_deaths'] = conc_data_row_new['deaths_y'] - conc_data_row_new['deaths_x']
conc_data_row_new.drop(['confirmed_y', 'deaths_y'], axis = 1, inplace = True)
conc_data_row_new.rename(columns = {'confirmed_x':'confirmed','deaths_x':'deaths'}, inplace = True)
conc_data_row_new.insert(3,'month', pd.DatetimeIndex(conc_data_row_new.date).month)
#conc_data_row_new['month_d'] = conc_data_row_new['month'].map(month)
# Make a data frame
 
def plot_new_cases(df):    
    df=pd.DataFrame({'x': df.date, 'c':df.continent, 'y1': df.new_cases, 'y2': df.new_deaths})
    #plt.style.use('fivethirtyeight')
    plt.style.use('seaborn-darkgrid')
    my_dpi=60

    #(96/my_dpi, 480/my_dpi)
    
    #plt.figure(figsize=(10, 10), dpi=my_dpi)
    f = plt.figure(figsize=(8, 3))
    ax = f.add_subplot(111)

    palette = plt.get_cmap('Set1')
    font = {'family': 'arial',
            'color':  'darkred',
            'weight': 'normal',
            'size': 20
            }

    #Now re do the interesting curve, but biger with distinct color
    plt.plot(df['x'], df['y1'], marker='', color=palette(1), linewidth=4, alpha=0.7) 
    plt.plot(df['x'],df['y2'], marker='', color='grey', linewidth=1, alpha=0.4)

    
    monthyearFmt = mdates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(monthyearFmt)
    plt.yticks([])

    # Adding labes of total deaths
    x_pos = df.x.tail(1).item()+ timedelta(days=5)
    plt.text(x_pos, df.y2.tail(1), str(df.y2.sum()) + ' Total Deaths', horizontalalignment='left', size=12, color='grey', fontdict=font)
    
    #Adding information of last amount of new cases 
    plt.text(x_pos, df.y1.max()/2, str(int(df.y1.sum())) + ' Last new cases' , horizontalalignment='left', size=12, color=palette(1), fontdict=font)
    
    #Adding information such as the highet pick
    max_y1 = df.y1.max()
    day = df[df['y1']== max_y1]
    text =  str(max_y1) + ' Highest pick \nof new cases on '+ str(day.x.item().strftime('%d %b'))
    plt.text(x_pos, df.y1.max()*0.9, text , horizontalalignment='left', size=12, color=palette(1), fontdict=font)
    
    plt.title('Distribution of new cases: ' + df.c.max(), 
              loc='left', fontsize=16, fontweight=0, color='black', fontdict=font, pad = 10)
p_continent = conc_data_row_new.groupby(['continent', 'date'],as_index=False)['new_cases','new_deaths'].sum()

for i in p_continent.continent.unique():
    df = pd.DataFrame(p_continent[p_continent['continent'] == i])
    plot_new_cases(df)
temp_data = conc_data_row_new.copy()
temp_data = temp_data.groupby('date', as_index=False)['new_cases','new_deaths'].sum()
temp_data.sort_values('date', inplace = True)

# Plot
f = plt.figure(figsize=(15,10))
ax = f.add_subplot(111)

plt.plot(temp_data.date,temp_data.new_cases,color=palette(1),linewidth=2,  marker='o',markersize=5)

# Tick-Parameters
monthyearFmt = mdates.DateFormatter('%d %b')
ax.xaxis.set_major_formatter(monthyearFmt)

ax = plt.gca()
ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
plt.tick_params(labelsize = 13) 

# Plot Title
plt.title('Distribution of new cases over the time',loc='left', fontsize=18, color='black', pad = 10)


plt.show()

def barv_graph(df):
    
    #plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')

    my_dpi=100
    #plt.subplots(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
    f = plt.figure(figsize=(8,30))
    ax = plt.subplot(711)
    

    x_pos = np.arange(len(df))
    width = 0.3

    #plt.subplots(figsize = (10,8))
    ax.barh(x_pos, df.Active, width, color=palette(8), edgecolor='white', label = 'Active')
    ax.barh(x_pos+(width), df.Deaths, width, color='orange', edgecolor='white', label = 'Deaths')
    ax.barh(x_pos+(width*2), df.Confirmed, width, color=palette(1), edgecolor='white', label='Confirmed')

    text = '10 countries with most confirmed cases in '+ df.continent.max()
    plt.title(text , loc='left', fontsize = 18, pad = 10)
    ax.set(yticks=x_pos + (width*2), yticklabels=df.country, ylim=[2*width - 1, len(df)])
    #plt.yticks(x_pos, df.country)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    plt.tick_params(labelsize = 12)   
    
    ax.legend()
    plt.show()
    
    
df_p = df_countries_cases.groupby(['country','continent'], as_index = False)['Confirmed', 'Deaths', 'Recovered','Active'].sum()

for x in df_p.continent.unique():
    df = df_p[df_p['continent']==x]
    df.sort_values('Confirmed',ascending = False, inplace = True)
    df= df.head(10)
    df.sort_values('Confirmed',ascending = True, inplace = True)
    barv_graph(df)
     