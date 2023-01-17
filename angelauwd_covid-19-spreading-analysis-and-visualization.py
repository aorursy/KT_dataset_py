# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from pandas import DataFrame

from datetime import datetime

import plotly.express as px

import plotly.graph_objects as go
#Data Resource: Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE,https://github.com/CSSEGISandData/COVID-19

#Orginal version: 2/15/2020

#Version updated: 2/26/2020

    #Added latest confirm map(used iso 3-letter country code)

    #Added animation on confirm by regions bar charts

    #Changed maps' visual designs, rearranged color scales

#Version updated: 3/14/2020

    #Added User Interaction Program

    #Added US COVID-19 Map 

#Version updated: 3/26/2020

    #Complied with source data changes

    #Resigned interactive program
confirmed_raw = pd.read_excel('/kaggle/input/time_series_covid19_confirmed_global.xlsx').fillna(0)

death_raw = pd.read_excel('/kaggle/input/time_series_covid19_deaths_global.xlsx').fillna(0)

recovered_raw = pd.read_excel('/kaggle/input/time_series_covid19_recovered_global.xlsx').fillna(0)

geodata = pd.read_csv('/kaggle/input/geocodedata.csv')
confirmed = pd.merge(confirmed_raw, geodata, how = 'left').dropna()

death = pd.merge(death_raw, geodata, how = 'left').dropna()

recovered = pd.merge(recovered_raw, geodata, how = 'left').dropna()
cols = list(confirmed.columns)

cols = cols[:4]+[cols[-1]]+cols[4:-1]

confirmed = confirmed[cols]

death = death[cols]

recovered = recovered[cols]
#convert column name to datetime format

confirmedtimeframe = confirmed.iloc[:,5:len(confirmed.columns)]

confirmedtimeframe.columns = pd.to_datetime(confirmedtimeframe.columns).date

deathtimeframe = death.iloc[:,5:len(death.columns)]

deathtimeframe.columns = pd.to_datetime(deathtimeframe.columns).date

recoveredtimeframe = recovered.iloc[:,5:len(recovered.columns)]

recoveredtimeframe.columns = pd.to_datetime(recoveredtimeframe.columns).date
confirmed = pd.concat([confirmed.iloc[:,0:5],confirmedtimeframe],axis = 1,sort=True)

death = pd.concat([death.iloc[:,0:5], deathtimeframe],axis = 1,sort=True)

recovered = pd.concat([recovered.iloc[:,0:5], recoveredtimeframe],axis = 1,sort=True)
#Melt indivdual timeframe per region to 'Date' column

confirmed = confirmed.melt(id_vars=["Province/State", "Country/Region",'Lat','Long','Code'], var_name="Date", value_name="Confirmed")

death = death.melt(id_vars=["Province/State", "Country/Region",'Lat','Long','Code'], var_name="Date", value_name="Death")

recovered = recovered.melt(id_vars=["Province/State", "Country/Region",'Lat','Long','Code'], var_name="Date", value_name="Recovered")
#Consolidate dataframe that's ready to be used

data = pd.concat([confirmed, death.iloc[:,6],recovered.iloc[:,6]], axis =1, sort = True)
#separate China region from and other regions worldwide

chinaregion = data.loc[(data['Country/Region'] == 'China')|(data['Country/Region'] == 'Taiwan')|

                                     (data['Country/Region'] =='Hong Kong')|(data['Country/Region'] =='Macau'),:]

ind = data.loc[(data['Country/Region'] == 'China')|(data['Country/Region'] == 'Taiwan')|

                                     (data['Country/Region'] =='Hong Kong')|(data['Country/Region'] =='Macau'),:].index

nonchinaregion = data.drop(data.index[ind],inplace = False).reset_index(drop = True)
#chinaregionpivot = pd.pivot_table(chinaregion_copy,index=["Country/Region","Province/State",'Date'], 

#                                 values = ['Confirmed','Recovered','Death'],

#                                 aggfunc = np.mean)
#Returns all countries affected as dictionaries

Countries_affected = list(data['Country/Region'].unique())
#Convert the list of countries to uppercase 

countries_affected_uc = list(map(lambda x: x.upper(), Countries_affected))
#Aggregate the number of cases per country/region

worldregion = pd.DataFrame(data.groupby(['Country/Region','Date'])['Date','Confirmed','Recovered','Death'].agg('sum')).reset_index()

worldregion_copy = worldregion.copy()

worldregion_copy['Country/Region'] = worldregion_copy['Country/Region'].str.upper() 
#Function to extract timeframe table per location

#region: worldregion (original)/worldregion_copy(uppercases)

#country: Country/Region in question



def getlocationdata(region, country):

    location = region.loc[region['Country/Region'] == country].reset_index()

    location['Day'] = np.arange(1,len(location)+1,1)

    return location
#Function to get the incremental cases each day per location

#region: worldregion (original)/worldregion_copy(uppercases)

#country: Country/Region in question



def getincremental(region, country):

    incc = []

    incd = []

    incr = []

    for i in range(len(getlocationdata(region, country))-1):

        c = getlocationdata(region, country)['Confirmed'][i+1]-getlocationdata(region, country)['Confirmed'][i]

        d = getlocationdata(region, country)['Death'][i+1]-getlocationdata(region, country)['Death'][i]

        r = getlocationdata(region, country)['Recovered'][i+1]-getlocationdata(region, country)['Recovered'][i]

        incc.append(c)

        incd.append(d)

        incr.append(r)



    dic = {'Incremental Confirmed': incc, 'Incremental Death': incd, 'Incremental Recovered': incr}



    incremental = pd.DataFrame(data = dic)

    incremental['Day'] = np.arange(1,len(incremental)+1,1)

    return incremental
#Function of ploting the line chart combining total confirms and incremental confirms in each location

def getconfirmeddata(confirmtable, incrementalctable, country): 

    fig = go.Figure()

    #plot the total confirms trend

    fig.add_trace(go.Scatter(x = confirmtable['Day'], y=confirmtable['Confirmed'], name='Confirmed',

                             mode='lines+markers',line=dict(color='royalblue', width=2, dash='dash')))

    #plot the incremental confirms trend

    fig.add_trace(go.Scatter(x = incrementalctable['Day'], y=incrementalctable['Incremental Confirmed'], name='Incremental Confirmed',

                             mode='lines+markers',line=dict(color='rgb(153,204,255)', width=2, dash='dash')))

    fig.update_layout(title='Confirms in ' + country,

                           xaxis_title='Days',

                           yaxis_title='Confirms')

    fig.show()
#Function of ploting the line chart combining total death and incremental death in each location

def getdeathdata(deathtable, incrementaldtable, country): 

    fig = go.Figure()

    #plot the total death trend

    fig.add_trace(go.Scatter(x = deathtable['Day'], y=deathtable['Death'],

                        name='Death', mode='lines+markers',line=dict(color='firebrick', width=2, dash='dot')))

    #plot the incremental death trend

    fig.add_trace(go.Scatter(x = incrementaldtable['Day'], y=incrementaldtable['Incremental Death'], name='Incremental Death',

                             mode='lines+markers',line=dict(color='rgb(255,153,153)', width=2, dash='dash')))

    fig.update_layout(title='Death in ' + country,

                       xaxis_title='Days',

                       yaxis_title='Number of Patients')

    fig.show()
#Function to set up an internactive program that shows the confirm and death cases in the countries you'd like to see

#Inputs are limited to countries affected (not case sensitive)

#Can cease or restart the program anytime you needed

#Feel free to hover on the lines for the exact day and number of cases. (day,# of cases)



#def showgraph():

#    print('This little interactive program shows you the trend of confirms and deaths in your selected country in COVID-19 pandemic from 1/22/2020.\n')

#    def start_program():

#        selection = input('Type in Country/Region:')

#        if selection.upper() in countries_affected_uc:

#            location = getlocationdata(worldregion_copy, selection.upper())

#            incremental = getincremental(worldregion_copy, selection.upper())

#

#            print('Trend of Confirmed Cases:')

#            getconfirmeddata(location, incremental, selection.upper())

#            print('Trend of Death Cases:')

#            getdeathdata(location, incremental, selection.upper())

#        else:

#            print('Input not available, please try again.')

#            

#        restart = input("Would you like to restart this program?(y/n)")

#        if restart.lower() == 'y':

#            start_program()

#        if restart.lower() == "n":

#            print ("Ends program")

#    start_program() 

            
US_confirm = getlocationdata(worldregion_copy, 'US')

US_incremental = getincremental(worldregion_copy, 'US')

getconfirmeddata(US_confirm, US_incremental, 'US')
getdeathdata(US_confirm, US_incremental, 'US')
CHINA_confirm = getlocationdata(worldregion_copy, 'CHINA')

CHINA_incremental = getincremental(worldregion_copy, 'CHINA')

getconfirmeddata(CHINA_confirm, CHINA_incremental, 'CHINA')
getdeathdata(CHINA_confirm, CHINA_incremental, 'CHINA')
ITALY_confirm = getlocationdata(worldregion_copy, 'ITALY')

ITALY_incremental = getincremental(worldregion_copy, 'ITALY')

getconfirmeddata(ITALY_confirm, ITALY_incremental, 'ITALY')
getdeathdata(ITALY_confirm, ITALY_incremental, 'ITALY')
chinaregion['Date'] = pd.to_datetime(chinaregion['Date'])

chinaregion['Date'] = chinaregion['Date'].dt.strftime('%m.%d')
#Get the latest cases in China region

chinalatest = pd.DataFrame(chinaregion.groupby(['Province/State','Country/Region'])['Lat','Long','Code','Confirmed','Recovered','Death'].agg('max')).sort_values(by=['Confirmed'], ascending = False).reset_index()
#Plot latest confirms in China region

fig = px.bar(chinalatest, x='Province/State', y='Confirmed',color = 'Province/State',color_discrete_sequence= px.colors.qualitative.Set3,

             hover_data=['Confirmed'], title = 'Latest Confirmed in China Region')

fig.show()
#Plot latest confirms in China region except Hubei Province

fig = px.bar(chinalatest.loc[chinalatest['Province/State'] != 'Hubei',:], x='Province/State', y='Confirmed',color = 'Province/State',color_discrete_sequence= px.colors.qualitative.Set3,

             hover_data=['Confirmed'], title = 'Latest Confirmed in China Region (other than Hubei)')

fig.show()
#Shows the changes of confirmed cases since 1/22/20 in China region

#Click on 'Start' botton to start the animation

fig = px.bar(chinaregion, x='Province/State', y='Confirmed',color = 'Province/State',color_discrete_sequence= px.colors.qualitative.Set3,

             animation_frame="Date",animation_group="Province/State",hover_data=['Confirmed'], title = 'Confirmed in China Region')

fig.show()
#Shows the changes of confirmed cases since 1/22/20 in China region other than Hubei Province

#Click on 'Start' botton to start the animation

fig = px.bar(chinaregion.loc[chinaregion['Province/State'] != 'Hubei',:], x='Province/State', y='Confirmed',color = 'Province/State',

             color_discrete_sequence= px.colors.qualitative.Set3, animation_frame="Date",animation_group="Province/State",

             hover_data=['Confirmed'], title = 'Confirmed in China Region Excluding Hubei')

fig.show()
worldregion = pd.DataFrame(data.groupby(['Country/Region','Date'])['Date','Confirmed','Recovered','Death'].agg('sum')).reset_index()

worldlatest = pd.DataFrame(data.groupby(['Country/Region'])['Lat','Long','Code','Confirmed','Recovered','Death'].agg('max')).sort_values(by = 'Confirmed', ascending = False).reset_index()
worldregion_copy = worldregion.loc[worldregion['Confirmed'] >= 0,:]
worldregion_copy = worldregion_copy.sort_values(by = ['Date', 'Confirmed'], ascending = True)
worldregion_copy['Date'] = pd.to_datetime(worldregion_copy['Date'])

worldregion_copy['Date'] = worldregion_copy['Date'].dt.strftime('%m.%d')
#Plot the latest worldwide confirms 

fig = px.bar(worldlatest.loc[worldlatest['Confirmed'] >=500,:], x='Country/Region', y='Confirmed',color = 'Country/Region',color_discrete_sequence= px.colors.qualitative.Set3,

             hover_data=['Confirmed'], title = 'Latest Confirmed Worldwide')

fig.show()
#Shows the changes of confirmed cases worldwide since 1/22/20

#Click on 'Start' botton to start the animation

#More information will show when you hover on the bars



fig = px.bar(worldregion_copy, x='Country/Region', y='Confirmed',color = 'Country/Region',

             color_discrete_sequence= px.colors.qualitative.Set3,animation_frame="Date",animation_group="Country/Region",

             hover_data=['Confirmed'], title = 'Worldwide Confirms')



fig.show()
chinalatest['Death Rate'] = chinalatest['Death']/chinalatest['Confirmed']

chinalatest['Recover Rate'] = chinalatest['Recovered']/chinalatest['Confirmed']

worldlatest['Death Rate'] = worldlatest['Death']/worldlatest['Confirmed']

worldlatest['Recover Rate'] = worldlatest['Recovered']/worldlatest['Confirmed']
#Death v.s. Confirmed in China region except Hubei Province

#Larger the size, higher the death rate



fig = px.scatter(chinalatest.loc[chinalatest['Province/State'] != 'Hubei',:], x="Confirmed", y="Death", size = 'Death Rate',

                 color="Province/State",color_discrete_sequence= px.colors.qualitative.Plotly,size_max=60,

                 title = 'Deaths in Confirmed in Other Part of China')

fig.show()
#Death v.s. Confirmed worldwide

#Larger the size, higher the death rate



fig = px.scatter(worldlatest, x="Confirmed", y="Death", size = 'Death Rate',

                 color="Country/Region",color_discrete_sequence= px.colors.qualitative.Plotly,size_max=60,

                 title = 'Deaths in Confirmed Worldwide')

fig.show()
#Spreading in all China regions since 1/22/2020

#Click on the 'Start' botton the start the animation



fig = px.scatter_geo(chinaregion, lat ="Lat", lon = 'Long',

                    color="Confirmed",size = 'Confirmed', animation_frame="Date",

                    hover_name="Province/State", size_max=20,

                    color_continuous_scale=[[0, 'rgb(255,160,122)'],

                             [0.01,"rgb(255,99,71)"], 

                             [0.02,"rgb(220,20,60)"],

                             [0.2,"rgb(178,34,34)"],

                             [0.6,"rgb(165,42,42)"],

                             [0.8,"rgb(139,0,0)"],

                             [1.0,"rgb(128,0,0)"]])

fig.update_geos(

    showcoastlines=True, coastlinecolor="rgb(153, 76, 0)",

    showland=True, landcolor="rgb(255, 204, 153)",

    showocean=True, oceancolor="White",

)

fig.update_layout(title='Confirmed in China thru Timeline')

fig.show()
worldregion = pd.DataFrame(data.groupby(['Country/Region','Date'])['Date','Lat','Long','Confirmed'].agg('sum')).reset_index()

worldregion['Date'] = pd.to_datetime(worldregion['Date'])

worldregion['Date'] = worldregion['Date'].dt.strftime('%m.%d')
worldregion.loc[worldregion['Country/Region'] == 'China','Lat'] = 30.9756

worldregion.loc[worldregion['Country/Region'] == 'China','Long'] = 112.2707

worldregion.loc[worldregion['Country/Region'] == 'United Kingdom','Lat'] = 55.3781

worldregion.loc[worldregion['Country/Region'] == 'United Kingdom','Long'] = -3.436

worldregion.loc[worldregion['Country/Region'] == 'Netherlands','Lat'] = 12.5186

worldregion.loc[worldregion['Country/Region'] == 'Netherlands','Long'] = -70.0358

worldregion.loc[worldregion['Country/Region'] == 'France','Lat'] = 46.2276

worldregion.loc[worldregion['Country/Region'] == 'France','Long'] = 2.2137

worldregion.loc[worldregion['Country/Region'] == 'Canada','Lat'] =49.2827

worldregion.loc[worldregion['Country/Region'] == 'Canada','Long'] = -123.1207

worldregion.loc[worldregion['Country/Region'] == 'Australia','Lat'] =-28.0167

worldregion.loc[worldregion['Country/Region'] == 'Australia','Long'] = 153.4
#Worldwide spreading since 1/22/2020

#Click on the 'Start' botton the start the animation



fig = px.scatter_geo(worldregion, 

                     lat ="Lat", lon = 'Long',

                    color="Confirmed",size = 'Confirmed', animation_frame="Date",

                    hover_name="Country/Region", size_max=30,

                    color_continuous_scale=[[0, 'rgb(255,160,122)'],

                             [0.01,"rgb(255,99,71)"], 

                             [0.02,"rgb(220,20,60)"],

                             [0.2,"rgb(178,34,34)"],

                             [0.6,"rgb(165,42,42)"],

                             [0.8,"rgb(139,0,0)"],

                             [1.0,"rgb(128,0,0)"]],)

fig.update_geos(

    showcoastlines=True, coastlinecolor="rgb(153, 76, 0)",

    showland=True, landcolor="rgb(255, 204, 153)",

    showocean=True, oceancolor="White",

)

fig.update_layout(title='Confirmed Worldwide thru Timeline')

fig.show()
latestbycode = pd.DataFrame(worldlatest.groupby(['Code','Country/Region'])['Country/Region','Code','Confirmed','Recovered','Death'].agg('sum')).sort_values(by=['Confirmed'], ascending = False).reset_index()
#Latest worldwide confirms (Based on 3-letter country codes)



fig = go.Figure(data=go.Choropleth(

                locations = latestbycode['Code'],

                z = latestbycode['Confirmed'],

                text = latestbycode['Country/Region'],

                colorscale = [[0, 'rgb(255,222,173)'],

                             [0.001,"rgb(255,99,71)"], 

                             [0.005,"rgb(165,42,42)"],

                             [0.1,"rgb(178,34,34)"],

                             [1.0,"rgb(128,0,0)"]],

                autocolorscale=False,

                reversescale=False,

                marker_line_color='darkgray',

                marker_line_width=0.5,

                colorbar_title = 'Confirmed',

))



fig.update_layout(title='Latest Worldwide Confirmed')



fig.show()