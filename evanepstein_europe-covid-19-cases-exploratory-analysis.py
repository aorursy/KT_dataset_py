# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px



eur_df_new = pd.read_csv('../input/covid19-cases-in-africa/covid19_europe.csv')

#import europe Cases
eur_df_new.head()
eur_df_new.shape

#32138 records, 7 columns. Now we can dive more into the columns and their contents.
eur_df_new.dtypes

#data types
pd.unique(eur_df_new['Country_Region'])

#this dataset looks at europe
eur_df_new.isnull().sum()

#Missing Value Count.

#7515 states or provinces within a country missing here., 24 active cases missing.

eur_df_new['Province_State'].isnull().sum()/25682

#This depicts the percentage of the Province_States values that are missing. 

#The threshold I go by is that if upwards of 25-30% of the values are missing I drop the column.
eur_df_new = eur_df_new.drop(columns = 'Province_State')
eur_df_new.isnull().values.any()
eur_df_new = eur_df_new.dropna()
eur_df_new.shape #new shape
import datetime as dt

#use it to obtain month and year in column for potential grouping purposes


eur_df_new['ObservationDate'] = pd.to_datetime(eur_df_new['ObservationDate'])

eur_df_new['mnth_yr'] = eur_df_new['ObservationDate'].apply(lambda x: x.strftime('%m-%Y')) 



#change datetime format

eur_df_new.dtypes

#new data types, the datetime conversion was successful
eur_df_new.head()

eur_df_new = eur_df_new.sort_values(by = 'mnth_yr', ascending=True)



eur_df_new

#new column entry successful
eur_df_new[['new_confirmed','new_active','new_deaths','new_recoveries']] = (eur_df_new.sort_values

                                                                            (by=['ObservationDate'], ascending=True)

                        .groupby(['Country_Region'])[['Confirmed','Active','Recovered','Deaths']].shift(1))





#eur_df_new['new_actives'] = eur_df_new['Active'] - eur_df_orig['Active']

#eur_df_new['new_recoveries'] = eur_df_new['Recovered'] - eur_df_orig['Recovered']

#eur_df_new['new_deaths'] = eur_df_new['Deaths']- eur_df_orig['Deaths']
eur_df_new.head(20)
eur_df_new.dtypes

#new types


cases_over_time = eur_df_new.groupby(['ObservationDate'])[['Confirmed','Active','Recovered','Deaths']].agg(np.sum)  

#cases over time by ObservationDate
cases_over_time.shape 

#total confirmed, active, recovered, deaths, now we can compare across


cases_over_time

#A table with cases Over time
cases_over_time.dtypes
#from numpy import nan

#cases_over_time = cases_over_time.replace(nan,0)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

#matplotlib dateformatter

import matplotlib.figure as fig

import datetime as dt
cases_over_time.plot()

plt.title('Cases over Time')

plt.yscale('linear')

plt.xlabel('Dates')

plt.ylabel('Total Cases')

#the number of confirmed cases and recovered cases in europe have increased. 

#non interactive plot to look at our dataframe and see the trends of case changes over time.



cases_ot = eur_df_new.groupby('ObservationDate', as_index=False)[['Confirmed','Active','Recovered','Deaths']].agg(np.sum)



cases_ot.head()

cases_ot_long = pd.melt(cases_ot, id_vars = ['ObservationDate'], 

                        value_vars = ['Confirmed','Active','Recovered','Deaths'])

#need to change the data frame shape to use in plotly

cases_ot_long #new dataframe
import plotly.graph_objects as go

#import both plotly graph objects now as well as plotly express prior
cases_ot_long = cases_ot_long.rename(columns = {'ObservationDate':'Date', 'variable':'Case Action Recorded', 

                                                'value':'Total Cases'})

#renaming columns for plot
figure = px.line(cases_ot_long, x ='Date', y = 'Total Cases', color = 'Case Action Recorded', 

                 title = 'Case Changes Over Time')

figure.show()

#plotly express allows for interactive plots, now we can see the increase by day
cases_ot_no_confirmed = cases_ot_long[(cases_ot_long['Case Action Recorded']!='Confirmed')]

#only want to look at the makeup of confirmed cases: ex: the total % of confirmed cases that were recovered
px.pie(cases_ot_no_confirmed, values='Total Cases',names='Case Action Recorded',

       title='Confirmed Case Percentage Makeup')



#percentage of confirmed cases that are active, recovered, and have resulted in deaths.



total_overall_by_country = eur_df_new.groupby('Country_Region', as_index=False)[['Confirmed','Active','Recovered','Deaths']].sum()

#full table showing total confirmed,active,recovered,and deadly cases in each european country

total_overall_by_country
total_overall_by_country = total_overall_by_country.rename(columns={'Country_Region':'Country'})

#Renaming for Understanding purposes
total_overall_by_country = total_overall_by_country.set_index('Country')

#set index for y values
def plotly_dict(df):

    return {'z': df.values.tolist(),

            'x': df.columns.tolist(),

            'y': df.index.tolist()}

#creates a function in order to get the dataframe in the necessary dictionary format for a heatmap, 

#as well as specify the necessary arguments for plotly

         
fig = go.Figure(data=go.Heatmap(plotly_dict(total_overall_by_country)))

fig.show()

#heatmap of every country's covid cases and labeled. This is a start, want to potentially get 

#geographic JSON data in there to create an actual map