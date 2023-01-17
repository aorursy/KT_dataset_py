import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.dates as mdates

from matplotlib.ticker import ScalarFormatter



plt.style.use('fivethirtyeight')



import datetime

from datetime import timedelta



import geopandas as gpd

from geopandas.tools import geocode



import warnings

warnings.filterwarnings('ignore')



#Color

state = {'Active':'#8D90A1','Recovered':'#63B76C','Dead':'#ED0A3F'}
covid_clean = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                       parse_dates=['Date'])



sars_clean = pd.read_csv("../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv", 

                       parse_dates=['Date'])

sars_summary = pd.read_csv('../input/sars-outbreak-2003-complete-dataset/summary_data_clean.csv')



mers_country = pd.read_csv("../input/mers-outbreak-dataset-20122019/country_count_latest.csv")

mers_clean_weekly = pd.read_csv("../input/mers-outbreak-dataset-20122019/weekly_clean.csv")



mers_clean_weekly['Year'] = pd.to_datetime(mers_clean_weekly['Year'],format = '%Y')
#COVID 19



yearwise_grouping = covid_clean.groupby('Date')

covid_temp_df = yearwise_grouping['Confirmed','Deaths','Recovered','Active'].sum()



#covid_temp_df

fig, axes = plt.subplots(figsize=(8,6))



plt.plot(covid_temp_df['Confirmed'], color = '#E88E5A')

plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed'] ,alpha = 0.5, color = '#E88E5A')





#axes.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) #Works when not using plt.style

axes.xaxis.set_major_locator(mdates.MonthLocator())

axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.title('Total COVID Cases over Time')

plt.xlabel('Months (2020)')

plt.ylabel('Total Cases')

plt.xticks(rotation = None)

plt.show()
#SARS



yearwise_grouping = sars_clean.groupby('Date')

sars_temp_df = yearwise_grouping['Cumulative number of case(s)','Number of deaths','Number recovered'].sum()



#temp_df



fig, axes = plt.subplots(figsize = (8,6))

plt.plot(sars_temp_df['Cumulative number of case(s)'], color = '#02A4D3')

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Cumulative number of case(s)'] ,alpha = 0.2, color = '#02A4D3')



#axes.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) #Works when not using plt.style

axes.xaxis.set_major_locator(mdates.MonthLocator(interval = 1))

axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.title('Total SARS Cases over Time')

plt.xlabel('Months (2020)')

plt.ylabel('Total Cases')

plt.xticks(rotation = None)

plt.show()
#MERS



yearwise_grouping = mers_clean_weekly.groupby('Year')

mers_temp_df = yearwise_grouping['New Cases'].sum().to_frame()

mers_temp_df = mers_temp_df.expanding(min_periods = 1).sum()



#mers_temp_df



fig, axes = plt.subplots(figsize=(8,6))

plt.plot(mers_temp_df, color = '#A50B5E')

plt.fill_between(list(mers_temp_df.index),mers_temp_df['New Cases'] ,alpha = 0.2, color = '#A50B5E')



#axes.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) #Works when not using plt.style

axes.xaxis.set_major_locator(mdates.YearLocator())

axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.title('Total MERS Cases over Time')

plt.xlabel('Year')

plt.ylabel('Total Cases')

plt.xticks(rotation = None)

plt.show()
sec = ['#E88E5A', '#02A4D3', '#A50B5E']

sns.set_palette(sns.color_palette(sec))

mers_clean_weekly['Year'] = pd.to_datetime(mers_clean_weekly['Year'],format = '%Y')



formatter = ScalarFormatter()

formatter.set_scientific(False)



fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(16,14))



#FOR SUBPLOT[0]

axes[0]=plt.subplot(2,1,1)

sns.lineplot(y='Confirmed',x=covid_temp_df.index,data = covid_temp_df, ax = axes[0], label = 'COVID-19', palette = [sec])

sns.lineplot(y = 'Cumulative number of case(s)',x = sars_temp_df.index,data = sars_temp_df, ax = axes[0], palette = [sec], label = 'SARS')

sns.lineplot(y = 'New Cases', x = mers_temp_df.index,data = mers_temp_df, ax = axes[0], palette = [sec],label = 'MERS')



plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed'] ,alpha = 0.5, color = '#E88E5A')

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Cumulative number of case(s)'] ,alpha = 0.2, color = '#02A4D3')

plt.fill_between(list(mers_temp_df.index),mers_temp_df['New Cases'] ,alpha = 0.2, color = '#A50B5E')



plt.yscale('linear')



axes[0].xaxis.set_major_locator(mdates.YearLocator())

axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[0].ticklabel_format(useOffset=False, style='plain',axis = 'y')

axes[0].yaxis.set_major_formatter(formatter)



plt.title('Linear Scale for Total Cases')

plt.grid(True)



#FOR SUBPLOT[1]

axes[1]=plt.subplot(2,1,2)

sns.lineplot(y='Confirmed',x=covid_temp_df.index,data = covid_temp_df, ax = axes[1], 

             palette = [sec],label = 'COVID-19')

sns.lineplot(y = 'Cumulative number of case(s)',x = sars_temp_df.index,data = sars_temp_df, ax = axes[1], 

             palette = [sec], label = 'SARS')

sns.lineplot(y = 'New Cases', x = mers_temp_df.index,data = mers_temp_df, ax = axes[1], 

             palette = [sec],label = 'MERS')



plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed'] ,alpha = 0.5, color = '#E88E5A')

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Cumulative number of case(s)'] ,alpha = 0.2, color = '#02A4D3')

plt.fill_between(list(mers_temp_df.index),mers_temp_df['New Cases'] ,alpha = 0.2, color = '#A50B5E')



plt.yscale('log')



axes[1].xaxis.set_major_locator(mdates.YearLocator())

axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

#axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'y')

axes[1].yaxis.set_major_formatter(formatter)



plt.title('Log Scale for Total Cases')

plt.grid(True)



plt.show()
covid_temp_df['Confirmed_per_day']=covid_temp_df['Confirmed'].diff()

covid_temp_df['Confirmed_per_day'][0]=covid_temp_df['Confirmed'][0]



#covid_temp_df



fig, axes = plt.subplots(figsize=(8,6))



plt.plot(covid_temp_df['Confirmed_per_day'])

plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed_per_day'] ,alpha = 0.2,color = '#E88E5A')



#axes.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) #Works when not using plt.style

axes.xaxis.set_major_locator(mdates.MonthLocator())

axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.title('New COVID Cases per day')

plt.xlabel('Months (2020)')

plt.ylabel('Total Cases')

plt.xticks(rotation = None)

plt.show()
#SARS



sars_temp_df['Confirmed_per_day']=sars_temp_df['Cumulative number of case(s)'].diff()

sars_temp_df['Confirmed_per_day'][0]=sars_temp_df['Cumulative number of case(s)'][0]



#covid_temp_df



fig, axes = plt.subplots(figsize=(8,6))



plt.plot(sars_temp_df['Confirmed_per_day'], color = '#02A4D3')

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Confirmed_per_day'] ,alpha = 0.2, color = '#02A4D3')



#axes.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) #Works when not using plt.style

axes.xaxis.set_major_locator(mdates.MonthLocator())

axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.title('New SARS Cases per day')

plt.xlabel('Months (2020)')

plt.ylabel('Total Cases')

plt.xticks(rotation = None)

plt.show()
mers_temp_df['Confirmed_per_year'] = mers_temp_df['New Cases'].diff()

mers_temp_df['Confirmed_per_year'][0] = mers_temp_df['New Cases'][0]



#mers_temp_df



fig, axes = plt.subplots(figsize=(8,6))

plt.plot(mers_temp_df['Confirmed_per_year'], color = '#A50B5E')

plt.fill_between(list(mers_temp_df.index),mers_temp_df['Confirmed_per_year'] ,alpha = 0.2, color = '#A50B5E')



#axes.xaxis.set_minor_locator(mdates.DayLocator(interval=1)) #Works when not using plt.style

axes.xaxis.set_major_locator(mdates.YearLocator())

axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.title('New MERS Cases per Year')

plt.xlabel('Year')

plt.ylabel('Total Cases')

plt.xticks(rotation = None)

plt.show()
#Summary graph



fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(14,12))



#FOR SUBPLOT[0]

axes[0]=plt.subplot(2,1,1)

sns.lineplot(y='Confirmed_per_day',x=covid_temp_df.index,data = covid_temp_df, ax = axes[0], label = 'COVID-19')

sns.lineplot(y = 'Confirmed_per_day',x = sars_temp_df.index,data = sars_temp_df, ax = axes[0], palette = ['red'], label = 'SARS')

sns.lineplot(y = 'Confirmed_per_year', x = mers_temp_df.index,data = mers_temp_df, ax = axes[0], palette = ['orange'],label = 'MERS')



plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed_per_day'] ,alpha = 0.5, color = '#E88E5A')

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Confirmed_per_day'] ,alpha = 0.2, color = '#02A4D3')

plt.fill_between(list(mers_temp_df.index),mers_temp_df['Confirmed_per_year'] ,alpha = 0.2, color = '#A50B5E')



axes[0].xaxis.set_major_locator(mdates.YearLocator())

axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[0].ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.yscale('linear')

plt.title('Linear Scale for New Cases')

plt.grid(True)



#FOR SUBPLOT[1]

axes[1]=plt.subplot(2,1,2)

sns.lineplot(y='Confirmed_per_day',x=covid_temp_df.index,data = covid_temp_df, ax = axes[1], label = 'COVID-19')

sns.lineplot(y = 'Confirmed_per_day',x = sars_temp_df.index,data = sars_temp_df, ax = axes[1], palette = ['red'], label = 'SARS')

sns.lineplot(y = 'Confirmed_per_year', x = mers_temp_df.index,data = mers_temp_df, ax = axes[1], palette = ['orange'],label = 'MERS')





plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed_per_day'] ,alpha = 0.5, color = '#E88E5A')

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Confirmed_per_day'] ,alpha = 0.2, color = '#02A4D3')

plt.fill_between(list(mers_temp_df.index),mers_temp_df['Confirmed_per_year'] ,alpha = 0.2, color = '#A50B5E')



axes[1].xaxis.set_major_locator(mdates.YearLocator())

axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.yscale('log')

plt.title('Log Scale for New Cases')

plt.grid(True)



plt.show()
import math

#mers_clean_weekly['Week'].unique()



mers_clean_weekly['nth_week'] = mers_clean_weekly['Week']

to_check = mers_clean_weekly['nth_week'][0]

week = 0



for i in range(mers_clean_weekly.shape[0]):

    

    df_week = mers_clean_weekly['nth_week']

    

    if(i==0):

        to_check = df_week[0]

        df_week[i] = 0

    else:

        if(df_week[i]==to_check):

            df_week[i] = week

        else:

            to_check = df_week[i]

            week+=1

            df_week[i] = week



covid_temp_df['nth_week'] = covid_temp_df.index

sars_temp_df['nth_week'] = sars_temp_df.index



week = 0

for i in range(0,math.ceil(covid_temp_df.shape[0]),7):

    covid_temp_df['nth_week'][i:i+7] = week

    week+=1



week = 0

for i in range(0,math.ceil(sars_temp_df.shape[0]),7):

    sars_temp_df['nth_week'][i:i+7] = week

    week+=1
weekwise_grouping = mers_clean_weekly.groupby('nth_week')

mers_week_temp_df = weekwise_grouping.sum()['New Cases'].to_frame()



mers_week_temp_df['Cummulative Cases'] = mers_week_temp_df['New Cases']

mers_week_temp_df['Cummulative Cases'] = mers_week_temp_df['Cummulative Cases'].expanding(min_periods = 1).sum()



covid_week_temp_df = covid_temp_df.groupby('nth_week')

covid_week_temp_df = covid_week_temp_df['Confirmed_per_day'].sum().to_frame()

covid_week_temp_df.rename(columns = {'Confirmed_per_day':'Confirmed_per_week'},inplace = True)



sars_week_temp_df = sars_temp_df.groupby('nth_week')

sars_week_temp_df = sars_week_temp_df['Confirmed_per_day'].sum().to_frame()

sars_week_temp_df.rename(columns = {'Confirmed_per_day':'Confirmed_per_week'},inplace = True)
'''

fig, axes = plt.subplots(figsize=(10,6))



sns.lineplot(y = 'Confirmed_per_week', x = covid_week_temp_df.index,data = covid_week_temp_df, ax = axes, label = 'COVID-19')

sns.lineplot(y = 'Confirmed_per_week', x = sars_week_temp_df.index,data = sars_week_temp_df, ax = axes, palette = ['red'], label = 'SARS')

sns.lineplot(y = 'New Cases', x = mers_week_temp_df.index,data = mers_week_temp_df, ax = axes, palette = ['orange'],label = 'MERS')



#axes.xaxis.set_major_locator(mdates.YearLocator())

#axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



#plt.xticks(list(mers_week_temp_df.index + 1))

plt.yscale('linear')

plt.title('Linear Scale for New Cases')

plt.show()

'''
fig, axes = plt.subplots(nrows = 2, ncols = 1,figsize=(18,14))



axes[0] = plt.subplot(2,1,1)

sns.lineplot(y = covid_week_temp_df['Confirmed_per_week'][:19], x = covid_week_temp_df.index[:19],data = covid_week_temp_df, ax = axes[0], label = 'COVID-19')

sns.lineplot(y = sars_week_temp_df['Confirmed_per_week'][:19], x = sars_week_temp_df.index[:19],data = sars_week_temp_df, ax = axes[0], palette = ['red'], label = 'SARS')

sns.lineplot(y = mers_week_temp_df['New Cases'][:19], x = mers_week_temp_df.index[:19],data = mers_week_temp_df, ax = axes[0], palette = ['orange'],label = 'MERS')



plt.fill_between(list(covid_week_temp_df.index[:19]),covid_week_temp_df['Confirmed_per_week'][:19] ,alpha = 0.5, color = '#E88E5A')

plt.fill_between(list(sars_week_temp_df.index[:19]),sars_week_temp_df['Confirmed_per_week'][:19] ,alpha = 0.2, color = '#02A4D3')

plt.fill_between(list(mers_week_temp_df.index[:19]),mers_week_temp_df['New Cases'][:19] ,alpha = 0.2, color = '#A50B5E')



#axes.xaxis.set_major_locator(mdates.YearLocator())

#axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[0].ticklabel_format(useOffset=False, style='plain',axis = 'y')

axes[0].set_ylim([-10000,830000])

axes[0].yaxis.set_major_formatter(formatter)



plt.xticks([i for i in range(1,26)])

plt.yscale('linear')

plt.title('Linear Scale for New Cases')



axes[1] = plt.subplot(2,1,2)



sns.lineplot(y = covid_week_temp_df['Confirmed_per_week'][:19], x = covid_week_temp_df.index[:19],data = covid_week_temp_df, ax = axes[1], label = 'COVID-19')

sns.lineplot(y = sars_week_temp_df['Confirmed_per_week'][:19], x = sars_week_temp_df.index[:19],data = sars_week_temp_df, ax = axes[1], palette = ['red'], label = 'SARS')

sns.lineplot(y = mers_week_temp_df['New Cases'][:19], x = mers_week_temp_df.index[:19],data = mers_week_temp_df, ax = axes[1], palette = ['orange'],label = 'MERS')



plt.fill_between(list(covid_week_temp_df.index[:19]),covid_week_temp_df['Confirmed_per_week'][:19] ,alpha = 0.5, color = '#E88E5A')

plt.fill_between(list(sars_week_temp_df.index[:19]),sars_week_temp_df['Confirmed_per_week'][:19] ,alpha = 0.2, color = '#02A4D3')

plt.fill_between(list(mers_week_temp_df.index[:19]),mers_week_temp_df['New Cases'][:19] ,alpha = 0.2, color = '#A50B5E')



plt.yscale('symlog')

#axes.xaxis.set_major_locator(mdates.YearLocator())

#axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[1].yaxis.set_major_formatter(formatter)



axes[1].set_ylim([-1000,100000000])

plt.xticks([i for i in range(1,20)])

plt.title('Log Scale for New Cases')



plt.show()
#Adding nth day and year



covid_temp_df['nth_day'] = covid_temp_df.index

covid_temp_df['nth_day'] = (covid_temp_df['nth_day'] - min(covid_temp_df['nth_day'])).dt.days

covid_temp_df['nth_year'] = covid_temp_df.index

covid_temp_df['nth_year'] = (covid_temp_df['nth_year'].dt.year - min(covid_temp_df['nth_year']).year)



sars_temp_df['nth_day'] = sars_temp_df.index

sars_temp_df['nth_day'] = (sars_temp_df['nth_day'] - min(sars_temp_df['nth_day'])).dt.days

sars_temp_df['nth_year'] = sars_temp_df.index

sars_temp_df['nth_year'] = (sars_temp_df['nth_year'].dt.year - min(sars_temp_df['nth_year']).year)



mers_temp_df['nth_year'] = mers_temp_df.index

mers_temp_df['nth_year'] = (mers_temp_df['nth_year'].dt.year - min(mers_temp_df['nth_year']).year)
world_map = gpd.read_file('../input/human-development-index-hdi/countries.geojson')



world_map = world_map[['name','continent','geometry']]

world_map.head()
#COVID



covid_countries = covid_clean[covid_clean['Date'] == max(covid_clean['Date'])]

covid_countries.rename(columns = {'Country/Region':'Country'},inplace = True)



covid_countries['Country'] = covid_countries['Country'].replace('South Korea', 'Korea')

covid_countries['Country'] = covid_countries['Country'].replace('US', 'United States')

covid_countries['Country'] = covid_countries['Country'].replace('Taiwan*', 'Taiwan')

covid_countries['Country'] = covid_countries['Country'].replace('Antigua and Barbuda', 'Antigua and Barb.')

covid_countries['Country'] = covid_countries['Country'].replace('Bosnia and Herzegovina', 'Bosnia and Herz.')

covid_countries['Country'] = covid_countries['Country'].replace('Burma', 'Myanmar')

covid_countries['Country'] = covid_countries['Country'].replace('Central African Republic', 'Central African Rep.')

covid_countries['Country'] = covid_countries['Country'].replace('Dominican Republic', 'Dominican Rep.')

covid_countries['Country'] = covid_countries['Country'].replace('Equatorial Guinea', 'Eq. Guinea')

covid_countries['Country'] = covid_countries['Country'].replace('Laos', 'Lao PDR')

covid_countries['Country'] = covid_countries['Country'].replace('Saint Kitts and Nevis', 'St. Kitts and Nevis')

covid_countries['Country'] = covid_countries['Country'].replace('Saint Vincent and the Grenadines', 'St. Vin. and Gren.')

covid_countries['Country'] = covid_countries['Country'].replace('Western Sahara', 'W. Sahara')

covid_countries['Country'] = covid_countries['Country'].replace('Cabo Verde', 'Cape Verde')

covid_countries['Country'] = covid_countries['Country'].replace('Congo (Kinshasa)', 'Dem. Rep. Congo')

covid_countries['Country'] = covid_countries['Country'].replace('Sao Tome and Principe', 'São Tomé and Principe')

covid_countries['Country'] = covid_countries['Country'].replace('Eswatini', 'Swaziland')

covid_countries['Country'] = covid_countries['Country'].replace("Cote d'Ivoire", "Côte d'Ivoire")

covid_countries['Country'] = covid_countries['Country'].replace('South Sudan', 'S. Sudan')

covid_countries['Country'] = covid_countries['Country'].replace('Czechia', 'Czech Rep.')

covid_countries['Country'] = covid_countries['Country'].replace('North Macedonia', 'Macedonia')

covid_countries['Country'] = covid_countries['Country'].replace('Congo (Brazzaville)', 'Congo')



covid_map = world_map.merge(covid_countries, left_on='name', right_on='Country')[['name','continent','Confirmed','geometry','Deaths']]
fig, ax = plt.subplots(figsize=(18, 8))



world_map.plot(ax=ax, color='lightgrey')

covid_map.plot(ax=ax, cmap='Oranges')

ax.set_facecolor('white')



covid_map[covid_map['Confirmed'] > 0].plot(ax=ax, cmap='Oranges', markersize=1)



plt.title("COVID confirmed in the world (DARKER means greater)")

plt

plt.xticks([])

plt.yticks([])

plt.show()
sars_summary = pd.read_csv('../input/sars-outbreak-2003-complete-dataset/summary_data_clean.csv')
#SARS



sars_summary['Country/Region'] = sars_summary['Country/Region'].replace('Hong Kong SAR, China', 'Hong Kong')

sars_summary['Country/Region'] = sars_summary['Country/Region'].replace('Macao SAR, China', 'Macao')

sars_summary['Country/Region'] = sars_summary['Country/Region'].replace('Republic of Ireland', 'Ireland')

sars_summary['Country/Region'] = sars_summary['Country/Region'].replace('Republic of Korea', 'Korea')

sars_summary['Country/Region'] = sars_summary['Country/Region'].replace('Russian Federation', 'Russia')

sars_summary['Country/Region'] = sars_summary['Country/Region'].replace('Taiwan, China', 'Taiwan')

sars_summary['Country/Region'] = sars_summary['Country/Region'].replace('Viet Nam', 'Vietnam')



sars_map = world_map.merge(sars_summary, left_on='name', right_on='Country/Region')[['name','continent','geometry','Cumulative total cases','No. of deaths']]
fig, ax = plt.subplots(figsize=(18, 8))



world_map.plot(ax=ax, color='lightgrey')

sars_map.plot(ax=ax, cmap='Blues')



sars_map[sars_map['Cumulative total cases'] > 0].plot(ax=ax, cmap='Blues', markersize=1)



#mers_map[mers_map['Confirmed'] > 100].plot(ax=ax, cmap='autumn', markersize=10)

#mers_map[mers_map['Confirmed'] > 1000].plot(ax=ax, cmap='autumn', markersize=10)



plt.title("SARS confirmed in the world")

plt.xticks([])

plt.yticks([])

plt.show()
mers_country = pd.read_csv('../input/mers-outbreak-dataset-20122019/country_count_latest.csv')
#MERS



mers_country['Country'] = mers_country['Country'].replace('Republic of Korea', 'Korea')

mers_country['Country'] = mers_country['Country'].replace('United States of America', 'United States')



mers_map = world_map.merge(mers_country, left_on='name', right_on='Country')

# mers_map
fig, ax = plt.subplots(figsize=(18, 8))



world_map.plot(ax=ax, color='lightgrey')

mers_map.plot(ax=ax, cmap='Purples')

ax.set_facecolor('white')



mers_map[mers_map['Confirmed'] > 0].plot(ax=ax, cmap='Purples', markersize=1)



#mers_map[mers_map['Confirmed'] > 100].plot(ax=ax, cmap='autumn', markersize=10)

#mers_map[mers_map['Confirmed'] > 1000].plot(ax=ax, cmap='autumn', markersize=10)



plt.title("MERS confirmed in the world")

plt.xticks([])

plt.yticks([])

plt.show()
#COVID



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(14, 12))



axes[0]=plt.subplot(2,1,1)



world_map.plot(ax=axes[0], color='lightgrey')

covid_map.plot(ax=axes[0], cmap='Oranges')

axes[0].set_facecolor('white')

plt.xticks([])

plt.yticks([])



covid_map[covid_map['Confirmed'] > 0].plot(ax=axes[0], cmap='Oranges', markersize=1)



#mers_map[mers_map['Confirmed'] > 100].plot(ax=ax, cmap='autumn', markersize=10)

#mers_map[mers_map['Confirmed'] > 1000].plot(ax=ax, cmap='autumn', markersize=10)



plt.title("COVID-19 confirmed in the world")



axes[1]=plt.subplot(2,1,2)

sorted_covid_by_country = covid_map.sort_values('Confirmed', ascending = False)

sns.barplot(x = sorted_covid_by_country['Confirmed'][:10], y = sorted_covid_by_country['name'][:10],ax = axes[1])



axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'x')

plt.title("Most affected Countries")

plt.show()
#SARS



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(14, 12))



axes[0]=plt.subplot(2,1,1)



world_map.plot(ax=axes[0], color='lightgrey')

sars_map.plot(ax=axes[0], cmap='Blues_r')

axes[0].set_facecolor('white')

plt.xticks([])

plt.yticks([])



sars_map[sars_map['Cumulative total cases'] > 0].plot(ax=axes[0], cmap='Blues_r', markersize=1)



#mers_map[mers_map['Confirmed'] > 100].plot(ax=ax, cmap='autumn', markersize=10)

#mers_map[mers_map['Confirmed'] > 1000].plot(ax=ax, cmap='autumn', markersize=10)



plt.title("SARS confirmed in the world")



axes[1]=plt.subplot(2,1,2)

sorted_sars_by_country = sars_map.sort_values('Cumulative total cases', ascending = False)

sns.barplot(x = sorted_sars_by_country['Cumulative total cases'][:10], y = sorted_sars_by_country['name'][:10],ax = axes[1])

plt.title("Most affected Countries")

plt.show()
#MERS



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(14, 12))



axes[0]=plt.subplot(2,1,1)



world_map.plot(ax=axes[0], color='lightgrey')

mers_map.plot(ax=axes[0], cmap='Purples')

axes[0].set_facecolor('white')



mers_map[mers_map['Confirmed'] > 0].plot(ax=axes[0], cmap='Purples', markersize=1)



plt.xticks([])

plt.yticks([])



plt.title("MERS confirmed in the world")



axes[1]=plt.subplot(2,1,2)

sorted_mers_by_country = mers_map.sort_values('Confirmed', ascending = False)

sns.barplot(x = sorted_mers_by_country['Confirmed'][:10], y = sorted_mers_by_country['name'][:10],ax = axes[1])

plt.title("Most affected Countries")

plt.show()
#mers_map['name'].nunique()

#covid_map['name'].nunique()

#sars_map['name'].nunique()



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(22, 8))



axes[0]=plt.subplot(1,2,1)

sns.barplot(y = [covid_map['name'].nunique(),mers_map['name'].nunique(),sars_map['name'].nunique()],

            x = ['COVID-19','MERS','SARS'], palette=sec)



plt.title('Number of Infected countries')



axes[1]=plt.subplot(1,2,2)

plt.pie([covid_map['name'].nunique(),mers_map['name'].nunique(),sars_map['name'].nunique()],

        labels = ['COVID-19','MERS','SARS'],

        autopct = '%1.1f%%',

        startangle = -120,

        explode = [0.2,0,0],

        colors = sec

       )

plt.title("Percentage of countries infected")



plt.show()
#COVID



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(14, 12))



axes[0]=plt.subplot(2,1,1)



world_map.plot(ax=axes[0], color='lightgrey')

covid_map.plot(ax=axes[0], cmap='Oranges')

axes[0].set_facecolor('white')

plt.xticks([])

plt.yticks([])



covid_map[covid_map['Deaths'] > 0].plot(ax=axes[0], cmap='Oranges', markersize=1)



#mers_map[mers_map['Confirmed'] > 100].plot(ax=ax, cmap='autumn', markersize=10)

#mers_map[mers_map['Confirmed'] > 1000].plot(ax=ax, cmap='autumn', markersize=10)



plt.title("COVID-19 Deaths in the world")



axes[1]=plt.subplot(2,1,2)



linegraph = sns.lineplot(y = covid_temp_df['Confirmed'], x = covid_temp_df.index,ax = axes[1],label = 'Confirmed', color = sec[0])

sns.lineplot(y = covid_temp_df['Deaths'], x = covid_temp_df.index,ax = axes[1],label = 'Dead',color = '#ED0A3F')



linegraph.lines[1].set_linestyle("--")



plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed'] ,alpha = 0.2, color = sec[0])

plt.fill_between(list(covid_temp_df.index),covid_temp_df['Deaths'] ,alpha = 0.2, color = '#ED0A3F')



axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.legend()

plt.title("Confirmed Vs Dead")

plt.show()
#SARS



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(14, 12))



axes[0]=plt.subplot(2,1,1)



world_map.plot(ax=axes[0], color='lightgrey')

sars_map.plot(ax=axes[0], cmap='Blues')



sars_map[sars_map['No. of deaths'] > 0].plot(ax=axes[0], cmap='Blues', markersize=1)



plt.title("SARS Deaths in the world")



axes[1]=plt.subplot(2,1,2)



linegraph = sns.lineplot(y = sars_temp_df['Cumulative number of case(s)'], x = sars_temp_df.index,ax = axes[1], label = 'Confirmed', color = sec[1])

sns.lineplot(y = sars_temp_df['Number of deaths'], x = sars_temp_df.index,ax = axes[1], label = 'Dead',color = '#ED0A3F')

axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'y')



linegraph.lines[1].set_linestyle("--")



plt.fill_between(list(sars_temp_df.index),sars_temp_df['Cumulative number of case(s)'] ,alpha = 0.2, color = sec[1])

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Number of deaths'] ,alpha = 0.2,color = '#ED0A3F')



plt.legend()

plt.title("Most affected Countries")

plt.show()
#COVID



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(22, 8))



axes[0]=plt.subplot(1,2,1)



#mers_map[mers_map['Confirmed'] > 100].plot(ax=ax, cmap='autumn', markersize=10)

#mers_map[mers_map['Confirmed'] > 1000].plot(ax=ax, cmap='autumn', markersize=10)



plt.pie([covid_temp_df['Recovered'][-1],(covid_temp_df['Confirmed'][-1])-(covid_temp_df['Recovered'][-1])],

        startangle = 90,

        labels = ['Recovered','Dead/Active'],

        autopct = '%1.1f%%',

        explode = [0.2,0],

        colors = [state['Recovered'],state['Dead']]

       )



plt.title("COVID-19 Deaths in the world")



axes[1]=plt.subplot(1,2,2)



linegraph = sns.lineplot(y = covid_temp_df['Confirmed'], x = covid_temp_df.index,ax = axes[1], label = 'Confirmed',color = sec[0])

sns.lineplot(y = covid_temp_df['Recovered'], x = covid_temp_df.index,ax = axes[1], label = 'Recovered', color = state['Recovered'])

axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'y')



linegraph.lines[1].set_linestyle("--")



plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed'] ,alpha = 0.2,color = sec[0])

plt.fill_between(list(covid_temp_df.index),covid_temp_df['Recovered'] ,alpha = 0.5, color = state['Recovered'])



plt.legend()

plt.title("Confirmed vs Recovery")

plt.show()
#SARS



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(22, 8))



axes[0]=plt.subplot(1,2,1)



plt.pie([sars_temp_df['Number recovered'][-1],(sars_temp_df['Cumulative number of case(s)'][-1])-(sars_temp_df['Number recovered'][-1])],

        labels = ['Recovered','Dead/Active'],

        autopct = '%1.1f%%',

        startangle = 20,

        explode = [0.2,0],

        colors = [state['Recovered'],state['Dead']]

       )



plt.title("SARS Deaths in the world")



axes[1]=plt.subplot(1,2,2)



linegraph = sns.lineplot(y = sars_temp_df['Cumulative number of case(s)'], x = sars_temp_df.index,ax = axes[1], label = 'Confirmed', color = sec[1])

sns.lineplot(y = sars_temp_df['Number recovered'], x = sars_temp_df.index,ax = axes[1], label = 'Recovered', color = state['Recovered'])

axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'y')



linegraph.lines[1].set_linestyle("--")



plt.fill_between(list(sars_temp_df.index),sars_temp_df['Cumulative number of case(s)'] ,alpha = 0.2, color = sec[1])

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Number recovered'] ,alpha = 0.5, color = state['Recovered'])



plt.title("Confirmed vs Recovery")

plt.show()
#COVID



fig, axes = plt.subplots(figsize=(16, 8))

linegraph = sns.lineplot(y = covid_temp_df['Confirmed'], x = covid_temp_df.index,ax = axes, color = sec[0])

sns.lineplot(y = covid_temp_df['Active'], x = covid_temp_df.index,ax = axes, color = state['Active'])

sns.lineplot(y = covid_temp_df['Recovered'], x = covid_temp_df.index,ax = axes, color = state['Recovered'])

sns.lineplot(y = covid_temp_df['Deaths'], x = covid_temp_df.index,ax = axes, color = state['Dead'])





plt.fill_between(list(covid_temp_df.index),covid_temp_df['Confirmed'] ,alpha = 0.2, label = 'Confirmed', color = sec[0])

plt.fill_between(list(covid_temp_df.index),covid_temp_df['Active'] ,alpha = 0.3, label = 'Active', color = state['Active'])

plt.fill_between(list(covid_temp_df.index),covid_temp_df['Recovered'] ,alpha = 0.5, label = 'Recovered', color = '#9DE093')

plt.fill_between(list(covid_temp_df.index),covid_temp_df['Deaths'] ,alpha = 0.5, label = 'Dead',color = '#ED0A3F')





linegraph.lines[1].set_linestyle("--")

linegraph.lines[2].set_linestyle("--")

linegraph.lines[3].set_linestyle("--")



axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')

plt.legend()

plt.title("COVID-19 Summary")

plt.show()
col = ['Confirmed', 'Deaths', 'Recovered', 'Active']

values = [covid_temp_df['Confirmed'][-1],covid_temp_df['Deaths'][-1],

          covid_temp_df['Recovered'][-1],covid_temp_df['Active'][-1]]



fig, axes = plt.subplots(1, 4, figsize=(24, 4))

axes = axes.flatten()

fig.set_facecolor('white')



for ind, col in enumerate(col):

    axes[ind].text(0.5, 0.6, col, 

            ha='center', va='center',

            fontfamily='monospace', fontsize=32, 

            color='white', backgroundcolor='black')



    axes[ind].text(0.5, 0.3, int(values[ind]), 

            ha='center', va='center',

            fontfamily='monospace', fontsize=48, fontweight='bold',

            color=sec[0], backgroundcolor='white')



    axes[ind].set_axis_off()
#SARS



fig, axes = plt.subplots(figsize=(16, 8))

linegraph = sns.lineplot(y = sars_temp_df['Cumulative number of case(s)'], x = sars_temp_df.index,ax = axes, label = 'Confirmed',color = sec[1])

sns.lineplot(y = sars_temp_df['Number recovered'], x = sars_temp_df.index,ax = axes, label = 'Recovered', color = state['Recovered'])

sns.lineplot(y = sars_temp_df['Number of deaths'], x = sars_temp_df.index,ax = axes, label = 'Dead', color = state['Dead'])

axes.ticklabel_format(useOffset=False, style='plain',axis = 'y')



linegraph.lines[1].set_linestyle("--")

linegraph.lines[2].set_linestyle("--")



plt.fill_between(list(sars_temp_df.index),sars_temp_df['Cumulative number of case(s)'] ,alpha = 0.2, color = sec[1])

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Number recovered'] ,alpha = 0.2, color = '#88B04B')

plt.fill_between(list(sars_temp_df.index),sars_temp_df['Number of deaths'] ,alpha = 0.2,color = '#DD4124')



plt.legend()

plt.title("Most affected Countries")

plt.show()
col = ['Confirmed', 'Deaths', 'Recovered', 'Active']

values = [sars_temp_df['Cumulative number of case(s)'][-1],sars_temp_df['Number of deaths'][-1],

          sars_temp_df['Number recovered'][-1],0]



fig, axes = plt.subplots(1, 4, figsize=(24, 4))

axes = axes.flatten()

fig.set_facecolor('white')



for ind, col in enumerate(col):

    axes[ind].text(0.5, 0.6, col, 

            ha='center', va='center',

            fontfamily='monospace', fontsize=32, 

            color='white', backgroundcolor='black')



    axes[ind].text(0.5, 0.3, int(values[ind]), 

            ha='center', va='center',

            fontfamily='monospace', fontsize=48, fontweight='bold',

            color=sec[1], backgroundcolor='white')



    axes[ind].set_axis_off()
#MERS



fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(16, 8))



axes[0] = plt.subplot(1,2,1)

sns.lineplot(y = mers_week_temp_df['Cummulative Cases'], x = mers_week_temp_df.index,ax = axes[0], color = sec[2])

plt.fill_between(list(mers_week_temp_df.index),mers_week_temp_df['Cummulative Cases'] ,alpha = 0.2, color = sec[2])



axes[0].ticklabel_format(useOffset=False, style='plain',axis = 'y')

plt.title("MERS Confirmed over Weeks")

plt.xlabel("Weeks")



axes[1] = plt.subplot(1,2,2)



plt.plot(mers_temp_df['New Cases'], color = sec[2])

plt.fill_between(list(mers_temp_df.index),mers_temp_df['New Cases'] ,alpha = 0.2, color = sec[2])

axes[1].xaxis.set_major_locator(mdates.YearLocator())

axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))



axes[1].ticklabel_format(useOffset=False, style='plain',axis = 'y')



plt.title('MERS Confirmed over Years')

plt.xlabel('Years')

plt.ylabel('Total Cases')

plt.xticks(rotation = None)



plt.show()
col = ['Confirmed', 'Deaths', 'Recovered', 'Active']

values = [mers_temp_df['New Cases'][-1],858,

          2447 -858,0]



fig, axes = plt.subplots(1, 4, figsize=(24, 4))

axes = axes.flatten()

fig.set_facecolor('white')



for ind, col in enumerate(col):

    axes[ind].text(0.5, 0.6, col, 

            ha='center', va='center',

            fontfamily='monospace', fontsize=32, 

            color='white', backgroundcolor='black')



    axes[ind].text(0.5, 0.3, int(values[ind]), 

            ha='center', va='center',

            fontfamily='monospace', fontsize=48, fontweight='bold',

            color=sec[2], backgroundcolor='white')



    axes[ind].set_axis_off()
#COVID



fig, axes = plt.subplots(nrows = 1, ncols = 3,figsize=(22, 8))



axes[0]=plt.subplot(1,3,1)



plt.pie([covid_temp_df['Recovered'][-1],

         covid_temp_df['Deaths'][-1],covid_temp_df['Active'][-1]],

        labels = ['Recovered','Deaths','Active'],

        autopct = '%1.1f%%',

        startangle = 90,

        explode = [0.2,0,0],

        colors = [state['Recovered'],state['Dead'],state['Active']]

       )



plt.title("COVID-19 Confirmed Breakdown")



axes[1]=plt.subplot(1,3,2)



plt.pie([sars_temp_df['Number recovered'][-1],

        sars_temp_df['Number of deaths'][-1]],

        labels = ['Recovered','Death'],

        autopct = '%1.1f%%',

        startangle = 20,

        explode = [0.2,0],

        colors = [state['Recovered'],state['Dead']]

       )



plt.title("SARS Confirmed Breakdown")



axes[1]=plt.subplot(1,3,3)

plt.pie([2494 -858,858 ],

        labels = ['Recovered','Death'],

        autopct = '%1.1f%%',

        startangle = 50,

        explode = [0.2,0],

        colors = [state['Recovered'],state['Dead']]

       )



plt.title("MERS Confirmed Breakdown")

plt.show()
#BarChart for Raw Numbers

x_indexes = np.array(['Confirmed','Death','Recovered','Countries Infected'])

x_pos = np.array([1,2,3,4])

shift_width = 0.25



covid_val = [covid_temp_df['Confirmed'][-1],

             covid_temp_df['Deaths'][-1],

             covid_temp_df['Recovered'][-1],

             len(covid_map)]



sars_val = [sars_temp_df['Cumulative number of case(s)'][-1],

            sars_temp_df['Number of deaths'][-1],

            sars_temp_df['Number recovered'][-1],

            len(sars_map)]



mers_val = [mers_temp_df['New Cases'][-1],

            858,

            2494 -858,

            len(mers_map)]



def makebarplot(index, scale,title_sent):

    axes[index]=plt.subplot(1,2,index+1)



    plt.bar((x_pos-shift_width),covid_val,label = 'COVID-19',width = 0.25,color = sec[0])

    plt.bar((x_pos),sars_val,label = 'SARS',width = 0.25,color = sec[1])

    plt.bar((x_pos+shift_width),mers_val,label = 'MERS',width = 0.25,color = sec[2])



    plt.xticks(x_pos,x_indexes)

    plt.title(title_sent)

    plt.ylabel("Total Number")



    plt.yscale(scale)

    plt.legend()



    for axis in [axes[index].yaxis]:

        formatter = ScalarFormatter()

        formatter.set_scientific(False)

        axis.set_major_formatter(formatter)



fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(20,6))

        

makebarplot(0,'linear','Linear Scale : Comparision of Raw Numbers')

makebarplot(1,'log','Log Scale : Comparision of Raw Numbers')

plt.show()
#BarChart for Percentages

x_indexes = np.array(['Death %','Recovery %','Infection % \n(R0 Score)','Countries% Infected'])

x_pos = np.array([1,2,3,4])

shift_width = 0.25



covid_val_per = [(covid_temp_df['Deaths'][-1]/covid_temp_df['Confirmed'][-1])*100,

            (covid_temp_df['Deaths'][-1]/covid_temp_df['Confirmed'][-1])*100,

            (2.5/18)*100, covid_map['name'].nunique()/195 * 100]



sars_val_per = [(sars_temp_df['Number of deaths'][-1]/sars_temp_df['Cumulative number of case(s)'][-1])*100,

            (sars_temp_df['Number recovered'][-1]/sars_temp_df['Cumulative number of case(s)'][-1])*100,

            (4/18)*100,sars_map['name'].nunique()/195*100]



mers_val_per = [(858/mers_temp_df['New Cases'][-1])*100,

            ((2494-858)/mers_temp_df['New Cases'][-1])*100,

            (0.8/18)*100,mers_map['name'].nunique()/195*100]



def makebarplot(index, scale, title_sent):

    axes[index]=plt.subplot(1,2,index+1)



    plt.bar((x_pos-shift_width),covid_val_per,label = 'COVID-19',width = 0.25,color = sec[0])

    plt.bar((x_pos),sars_val_per,label = 'SARS',width = 0.25,color = sec[1])

    plt.bar((x_pos+shift_width),mers_val_per,label = 'MERS',width = 0.25,color = sec[2])



    plt.xticks(x_pos,x_indexes)

    plt.title(title_sent)

    plt.ylabel("% out of 100")



    plt.yscale(scale)

    plt.legend()

    



    for axis in [axes[index].yaxis]:

        formatter = ScalarFormatter()

        formatter.set_scientific(False)

        axis.set_major_formatter(formatter)

        

    axes[index].set_ylim([0,100])



fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(20,6))

        

makebarplot(0,'linear','Linear Scale : Comparision of Percentages')

makebarplot(1,'log','Log Scale : Comparision of Percentages')

plt.show()
#Raw Numbers

#HIGHEST CONFIRMED

#HIGHEST DEATH COUNT

#HIGHEST RECOVERIES

#HIGHEST COUNTRIES INFECTED
col = ['Confirmed', 'Deaths', 'Recovered', 'Countries Infected',]



fig, axes = plt.subplots(1, 4, figsize=(24, 4))

axes = axes.flatten()

fig.set_facecolor('white')



for ind, col in enumerate(col):

    axes[ind].text(0.5, 0.6, col, 

            ha='center', va='center',

            fontfamily='monospace', fontsize=32,

            color='white', backgroundcolor='black')



    axes[ind].text(0.5, 0.3, 'COVID-19', 

            ha='center', va='center',

            fontfamily='monospace', fontsize=48, fontweight='bold',

            color=sec[0], backgroundcolor='white')

    

    axes[ind].text(0.5, 0, int(covid_val[ind]), 

            ha='center', va='center',

            fontfamily='monospace', fontsize=48, fontweight='bold',

            color=sec[0], backgroundcolor='white')



    axes[ind].set_axis_off()
#Percentage



#HIGHEST DEATH CHNACE

#HIGHEST RECOVERY PERCENTAGE

#INFECTION INFECTIVITY



col = ['Death%', 'Recovery%', '(R0 Score)', 'Countries% Infected',]



values = [['MERS',mers_val_per[0]],['SARS',sars_val_per[1]],['SARS',sars_val_per[2]],['COVID-19',covid_val_per[3]]]

color_val = [sec[2],sec[1],sec[1],sec[0]]



fig, axes = plt.subplots(1, 4, figsize=(24, 4))

axes = axes.flatten()

fig.set_facecolor('white')



for ind, col in enumerate(col):

    axes[ind].text(0.5, 0.6, col, 

            ha='center', va='center',

            fontfamily='monospace', fontsize=32,

            color='white', backgroundcolor='black')



    axes[ind].text(0.5, 0.3, values[ind][0], 

            ha='center', va='center',

            fontfamily='monospace', fontsize=48, fontweight='bold',

            color=color_val[ind], backgroundcolor='white')

    

    axes[ind].text(0.5, 0, str(round(values[ind][1],2))+str('%'), 

            ha='center', va='center',

            fontfamily='monospace', fontsize=48, fontweight='bold',

            color=color_val[ind], backgroundcolor='white')



    axes[ind].set_axis_off()