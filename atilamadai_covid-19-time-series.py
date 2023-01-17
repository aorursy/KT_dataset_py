# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#!pip install us

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from pylab import rcParams

from pandas.api.types import CategoricalDtype

from datetime import date, timedelta

import warnings

import folium

from folium.plugins import HeatMap

import pandas_profiling

import math

import pickle

import fbprophet

from fbprophet.plot import plot_plotly

from fbprophet.plot import add_changepoints_to_plot

import glob

#import us



rcParams["figure.figsize"] = 20,9

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import data from github url, url1 -Confirmed, url2 -Deaths. url3 -Recovered

url1='/kaggle/input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

url2='/kaggle/input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

url3='/kaggle/input/covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

df1 = pd.read_csv(url1, error_bad_lines=False)

df2 = pd.read_csv(url2, error_bad_lines=False)

df3 = pd.read_csv(url3, error_bad_lines=False)

# display df1 -Confirmed df2 -Deaths df3 -Recovered

display(df1.head(), df2.head(), df3.head())
# Data Subset for US States

path = r'/kaggle/input/covid19/csse_covid_19_data/csse_covid_19_daily_reports/' # path to daily reports



all_files = glob.glob(path + "/*.csv")



li = []



for filename in all_files:



    df = pd.read_csv(filename, index_col=None, header=0)



    li.append(df)



frame = pd.concat(li, axis=0, ignore_index=True)

frame["Last Update"] = frame['Last Update'].fillna(frame['Last_Update'])

frame["Country/Region"] = frame['Country/Region'].fillna(frame['Country_Region'])

frame["Province/State"] = frame['Province/State'].fillna(frame['Province_State'])

frame['Last Update'] = pd.to_datetime(frame['Last Update'])

# Create filter for Last Update to equal LastUpdate

LastUpdateDaily=frame['Last Update'].max()

LastUpdate_FilterDaily=frame['Last Update']==LastUpdateDaily

TotalCasesByLastUpdateProvinceDaily=frame.groupby(['Country/Region', 'Province/State','Last Update'],sort=False).agg({'Confirmed':sum, 'Deaths':sum, 'Recovered':sum,'Active':sum})

TotalCasesByLastUpdateProvinceDaily=TotalCasesByLastUpdateProvinceDaily.reset_index()

TotalCasesByLastUpdateProvinceDaily['Last Update']= TotalCasesByLastUpdateProvinceDaily['Last Update'].dt.normalize()

TotalCasesByLastUpdateProvinceDaily=TotalCasesByLastUpdateProvinceDaily[TotalCasesByLastUpdateProvinceDaily['Country/Region']=='US'].reset_index()

#TotalCasesByLastUpdateProvinceDaily.to_excel(r'/kaggle/working/all1.xlsx', index = False)

#display(frame.head(), TotalCasesByLastUpdateProvinceDaily.head(), frame.info())
# unpivot data frames: df1 (Confirmed) and name it df1u, df2 (Deaths) and name it df2u, df3 (Recovered) and name it df3u

df1u=pd.melt(df1, id_vars=['Province/State', 'Country/Region', 'Lat','Long'], var_name='DateTime', value_name='Confirmed')

df2u=pd.melt(df2, id_vars=['Province/State', 'Country/Region', 'Lat','Long'], var_name='DateTime', value_name='Deaths')

df3u=pd.melt(df3, id_vars=['Province/State', 'Country/Region', 'Lat','Long'], var_name='DateTime', value_name='Recovered')

# show unpivoted data frame for Confirmed, Deaths, Recovered

display(df1u.head(), df2u.head(), df3u.head())
# Check number of NaN in data frame

display(df1u.info(), df2u.info(), df3u.info())
# fill NaN with 0 for Confirmed, Deaths, Recovered

df1u['Confirmed'] = df1u['Confirmed'].fillna(0)

df2u['Deaths'] = df2u['Deaths'].fillna(0)

df3u['Recovered'] = df3u['Recovered'].fillna(0)

# fill NaN in Province/State column with data from Country/Region column

df3u['Province/State'] = df3u['Province/State'].fillna(df3u['Country/Region'])

df2u['Province/State'] = df2u['Province/State'].fillna(df2u['Country/Region'])

df1u['Province/State'] = df1u['Province/State'].fillna(df1u['Country/Region'])

# Display fill NaN for Confirmed, Deaths, Recovered

display(df1u.head(), df2u.head(), df3u.head(), df1u.info(), df2u.info(), df3u.info())
# Change Data type for Columns DateTime, Confirmed, Deaths, and Recovered

df1u['DateTime'] = pd.to_datetime(df1u['DateTime'])

df2u['DateTime'] = pd.to_datetime(df2u['DateTime'])

df3u['DateTime'] = pd.to_datetime(df3u['DateTime'])

df1u['Confirmed']= pd.to_numeric(df1u['Confirmed'], downcast='integer')

df2u['Deaths']= pd.to_numeric(df2u['Deaths'], downcast='integer')

df3u['Recovered']= pd.to_numeric(df3u['Recovered'], downcast='integer')

display(df1u.info(), df2u.info(), df3u.info())
# merge data frame df1u, df2u, df3u

merged_df1= pd.merge(df1u, df2u, how='left', left_on=['Province/State','DateTime'], right_on=['Province/State','DateTime'])

all_data= pd.merge(merged_df1, df3u,how='left', left_on=['Province/State','DateTime'], right_on=['Province/State','DateTime'])

all_data.head(), all_data.info()
# Drop Colums 'Province/State_y', 'Country/Region_y', 'Province/State', 'Country/Region'



all_data.drop(columns=['Lat_y', 'Long_y', 'Lat', 'Long', 'Country/Region', 'Country/Region_y'], inplace=True)

all_data.head()
# Rename Columns Province/State_x to Province/State and Country/Region_x to Country/Region

all_data.rename(columns={"Lat_x": "Lat", "Long_x": "Long", "Country/Region_x": "Country/Region"}, inplace=True)

all_data['Confirmed'] = all_data['Confirmed'].fillna(0)

all_data['Deaths'] = all_data['Deaths'].fillna(0)

all_data['Recovered'] = all_data['Recovered'].fillna(0)

all_data['Recovered']= pd.to_numeric(all_data['Recovered'], downcast='integer')

all_data['Existing'] = all_data['Confirmed']-all_data['Deaths']-all_data['Recovered']

display(all_data.head(), all_data.info())
# Show Last Update of COVID-19 Cases

# Create filter for DateTime to equal LastUpdate

LastUpdate=all_data['DateTime'].max()

LastUpdate_Filter=all_data['DateTime']==LastUpdate
# Create filtered data frame by LastUpdate_Filter

TotalCases_LastUpdate=all_data[LastUpdate_Filter]

TotalCases_LastUpdate.sort_values(by=['Confirmed'], inplace=True, ascending=False)

TotalCases_LastUpdate.reset_index(0, drop=True, inplace=True)

TotalCases_LastUpdate['Province/State'] = TotalCases_LastUpdate['Province/State'].fillna(TotalCases_LastUpdate['Country/Region'])

TotalCases_LastUpdate['Existing'] = TotalCases_LastUpdate['Confirmed']-TotalCases_LastUpdate['Deaths']-TotalCases_LastUpdate['Recovered']

TotalConf=TotalCases_LastUpdate['Confirmed'].sum()

TotalDeaths=TotalCases_LastUpdate['Deaths'].sum()

TotalRecovered=TotalCases_LastUpdate['Recovered'].sum()

TotalExisting=TotalCases_LastUpdate['Existing'].sum()

NCountries=TotalCases_LastUpdate['Country/Region'].nunique()

EstMortalityRate=TotalDeaths/TotalConf

#display(TotalCases_LastUpdate.head(), TotalCases_LastUpdate.info(), TotalCases_PreviousDay.head(),TotalCasesDelta.head())

print("Last Update", LastUpdate.strftime('%d %b %Y'), sep=": ")

print("Total Confirmed", '{:,}'.format(TotalConf), sep=": ")

print("Total Deaths", '{:,}'.format(TotalDeaths), sep=": ")

print("Total Recovered", '{:,}'.format(TotalRecovered), sep=": ")

print("Existing",'{:,}'.format(TotalExisting), sep=": ")

print("Estimated Mortality rate",'{:.2%}'.format(EstMortalityRate), sep=": " )
# Total Cases by Country/Region and Last Update

TotalCasesByLastUpdateCountry=TotalCases_LastUpdate.groupby(['Country/Region'], sort=False)['Confirmed', 'Deaths', 'Recovered', 'Existing'].sum()

TotalCasesByLastUpdateCountry=TotalCasesByLastUpdateCountry.reset_index()

print("Last Update", LastUpdate.strftime('%d %b %Y'), sep=": ")

display(TotalCasesByLastUpdateCountry.style.background_gradient(cmap='Set1_r'))
# Total Cases by Country/Region, Province/State, and Last Update

TotalCasesByLastUpdateProvince=TotalCases_LastUpdate.groupby(['Country/Region', 'Province/State'],sort=False)['Confirmed', 'Deaths', 'Recovered','Existing'].sum()

TotalCasesByLastUpdateProvince=TotalCasesByLastUpdateProvince.reset_index()

print("Last Update", LastUpdate.strftime('%d %b %Y'), sep=": ")

display(TotalCasesByLastUpdateProvince.style.background_gradient(cmap='Set1_r'))
# Total Cases by Date

TotalCases=all_data.groupby(['DateTime']).agg({"Confirmed": sum, 'Deaths': sum, 'Recovered': sum, 'Existing': sum})

TotalCases=TotalCases.reset_index()

TotalCases['deltaConfirmed']=TotalCases['Confirmed'].diff().fillna(0)

TotalCases['deltaDeaths']=TotalCases['Deaths'].diff().fillna(0)

TotalCases['deltaRecovered']=TotalCases['Recovered'].diff().fillna(0)

TotalCases['deltaExisting']=TotalCases['Existing'].diff().fillna(0)

# Total Cases Excluding China by Date

ExChina=all_data[all_data['Country/Region']!='China']

ExChina=ExChina.reset_index()

TotalCasesExChina=ExChina.groupby(['DateTime']).agg({"Confirmed": sum, 'Deaths': sum, 'Recovered': sum, 'Existing': sum})

TotalCasesExChina=TotalCasesExChina.reset_index()

TotalCasesExChina['deltaConfirmed']=TotalCasesExChina['Confirmed'].diff().fillna(TotalCasesExChina['Confirmed'])

TotalCasesExChina['deltaDeaths']=TotalCasesExChina['Deaths'].diff().fillna(TotalCasesExChina['Deaths'])

TotalCasesExChina['deltaRecovered']=TotalCasesExChina['Recovered'].diff().fillna(TotalCasesExChina['Recovered'])

TotalCasesExChina['deltaExisting']=TotalCasesExChina['Existing'].diff().fillna(TotalCasesExChina['Existing'])

# Total Cases China by Date

China=all_data[all_data['Country/Region']=='China'].reset_index()

TotalCasesChina=China.groupby(['DateTime']).agg({"Confirmed": sum, 'Deaths': sum, 'Recovered': sum, 'Existing': sum}).reset_index()

TotalCasesChina['deltaConfirmed']=TotalCasesChina['Confirmed'].diff().fillna(TotalCasesChina['Confirmed'])

TotalCasesChina['deltaDeaths']=TotalCasesChina['Deaths'].diff().fillna(TotalCasesChina['Deaths'])

TotalCasesChina['deltaRecovered']=TotalCasesChina['Recovered'].diff().fillna(TotalCasesChina['Recovered'])

TotalCasesChina['deltaExisting']=TotalCasesChina['Existing'].diff().fillna(TotalCasesChina['Existing'])

# Total Cases by Province and Date

TotalCasesByProvince=all_data.groupby(['Province/State', 'DateTime']).agg({"Confirmed": sum, 'Deaths': sum, 'Recovered': sum, 'Existing': sum})

TotalCasesByProvince=TotalCasesByProvince.reset_index()

TotalCasesByProvince['Confirmed_Delta']=TotalCasesByProvince.sort_values(['Province/State', 'DateTime']).groupby('Province/State')['Confirmed'].diff().fillna(0)

TotalCasesByProvince['Deaths_Delta']=TotalCasesByProvince.sort_values(['Province/State', 'DateTime']).groupby('Province/State')['Deaths'].diff().fillna(0)

TotalCasesByProvince['Recovered_Delta']=TotalCasesByProvince.sort_values(['Province/State', 'DateTime']).groupby('Province/State')['Recovered'].diff().fillna(0)

TotalCasesByProvince['Existing_Delta']=TotalCasesByProvince.sort_values(['Province/State', 'DateTime']).groupby('Province/State')['Existing'].diff().fillna(0)

# Total Cases by Country and Date

TotalCasesByCountry=all_data.groupby(['Country/Region', 'DateTime']).agg({"Confirmed": sum, 'Deaths': sum, 'Recovered':sum, 'Existing': sum})

TotalCasesByCountry=TotalCasesByCountry.reset_index()

TotalCasesByCountry['Confirmed_Delta']=TotalCasesByCountry.sort_values(['Country/Region', 'DateTime']).groupby('Country/Region')['Confirmed'].diff().fillna(0)

TotalCasesByCountry['Deaths_Delta']=TotalCasesByCountry.sort_values(['Country/Region', 'DateTime']).groupby('Country/Region')['Deaths'].diff().fillna(0)

TotalCasesByCountry['Recovered_Delta']=TotalCasesByCountry.sort_values(['Country/Region', 'DateTime']).groupby('Country/Region')['Recovered'].diff().fillna(0)

TotalCasesByCountry['Existing_Delta']=TotalCasesByCountry.sort_values(['Country/Region', 'DateTime']).groupby('Country/Region')['Existing'].diff().fillna(0)



#display(TotalCases.head(), TotalCasesByProvince.head(), TotalCases.info(), TotalCasesByProvince.info())

cl=sns.light_palette("red", as_cmap=True)

display(TotalCases[['DateTime','Existing','Deaths','Recovered']].style.background_gradient(cmap=cl).set_caption('Heat Map of Total Existing, Deaths, and Recovered by Date'))

#display(TotalCasesByCountry[['Country/Region','DateTime','Existing','Deaths','Recovered']].style.background_gradient(cmap=cl).set_caption('Heat Map of Total Existing, Deaths, and Recovered by Country and Date'))

display(TotalCasesExChina.style.background_gradient(cmap=cl).set_caption('Heat Map Excluding China'))

display(TotalCasesChina.style.background_gradient(cmap=cl).set_caption('Heat Map China Cases'))

# Time Series Visual for Confirmed, Recovered, and Deaths 

fig = go.Figure()

fig.add_trace(go.Scatter(x = TotalCases.DateTime, y = TotalCases['Confirmed'], name = "Total Confirmed", line_color = 'orange', mode = 'lines+markers', marker = dict(size = 9, symbol = 'circle')))

fig.add_trace(go.Scatter(x = TotalCases.DateTime, y = TotalCases['Recovered'], name = "Total Recovered", line_color = 'green', mode = 'lines+markers', marker = dict(size = 9, symbol = 'circle')))

fig.add_trace(go.Scatter(x = TotalCases.DateTime, y = TotalCases['Deaths'], name = "Total Deaths", line_color = 'firebrick', mode = 'lines+markers', marker = dict(size = 9, symbol = 'circle')))

fig.update_layout(title_text = 'Time Series Total Cases',

  xaxis_rangeslider_visible = True,

  paper_bgcolor='rgba(233,233,233,233)',

    plot_bgcolor='rgba(0,0,0,0)')

fig.show()

# Time Series Visual for Existing Cases

fig = go.Figure()

fig.add_trace(go.Scatter(x = TotalCases.DateTime, y = TotalCases['Existing'], name = "Total Existing", line_color = 'blue', mode = 'lines+markers', marker = dict(size = 9, symbol = 'circle')))

fig.update_layout(title_text = 'Time Series Total Existing', showlegend=True,

  xaxis_rangeslider_visible = True,

  paper_bgcolor='rgba(233,233,233,233)',

    plot_bgcolor='rgba(0,0,0,0)')

fig.show()

# Time Series Visual for Existing Cases Except China

fig = go.Figure()

fig.add_trace(go.Scatter(x = TotalCasesExChina.DateTime, y = TotalCasesExChina['Existing'], name = "Total Existing Excluding China", line_color = 'violet', mode = 'lines+markers', marker = dict(size = 9, symbol = 'circle')))

fig.update_layout(title_text = 'Time Series Total Existing Excluding China', showlegend=True,

  xaxis_rangeslider_visible = True,

  paper_bgcolor='rgba(233,233,233,233)',

    plot_bgcolor='rgba(0,0,0,0)')

fig.show()

# Visual Display of Day to Day increase in new cases

fig = go.Figure()

fig.add_trace(go.Bar(x = TotalCases.DateTime, y = TotalCases['deltaConfirmed'], name = "Total Confirmed Delta", marker_color='orange') )

fig.add_trace(go.Bar(x = TotalCases.DateTime, y = TotalCases['deltaDeaths'], name = "Total Deaths Delta", marker_color='firebrick') )

fig.add_trace(go.Bar(x = TotalCases.DateTime, y = TotalCases['deltaRecovered'], name = "Total Recovered Delta", marker_color='green') )

fig.update_layout(title_text = 'Total Cases Delta',

  xaxis_rangeslider_visible = True,

  paper_bgcolor='rgba(233,233,233,233)',

    plot_bgcolor='rgba(0,0,0,0)')

fig.show()
# Day to Day increase in Confirmed Cases

fig = px.bar(TotalCasesByCountry, x="DateTime", y="Confirmed_Delta", color='Country/Region', orientation='v', height=600,

             title='Day to Day Increase in Confirmed Cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.update_layout(

  paper_bgcolor='rgba(233,233,233,233)',

    plot_bgcolor='rgba(0,0,0,0)')

fig.show()

# Day to Day increase in Recovered Cases

fig = px.bar(TotalCasesByCountry, x="DateTime", y="Recovered_Delta", color='Country/Region', orientation='v', height=600,

             title='Day to Day Increase in Total Recovered Cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.update_layout(

  paper_bgcolor='rgba(233,233,233,233)',

    plot_bgcolor='rgba(0,0,0,0)')

fig.show()

# Day to Day increase in Existing Cases

fig = px.bar(TotalCasesByCountry, x="DateTime", y="Existing_Delta", color='Country/Region', orientation='v', height=600,

             title='Day to Day Increase in Existing Cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.update_layout(

  paper_bgcolor='rgba(233,233,233,233)',

    plot_bgcolor='rgba(0,0,0,0)')

fig.show()

#Countries with confirmed cases

fig = px.choropleth(TotalCasesByLastUpdateCountry, locations="Country/Region", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country/Region", range_color=[1,10000], 

                    color_continuous_scale="orrd", 

                    title='Countries with Confirmed Cases')

fig.update_layout(

    paper_bgcolor='rgba(233,233,233,233)', 

    plot_bgcolor='rgba(0,0,0,0)', 

    coloraxis_showscale=False,

    title_x=0.5,

    margin={"r":0,"t":50,"l":0,"b":20})

fig.update_geos(resolution=110,

                projection_type="natural earth")

fig.show()

#US states by last update

USLastUpdateDaily=TotalCasesByLastUpdateProvinceDaily['Last Update'].max()

USLastUpdate_FilterDaily=TotalCasesByLastUpdateProvinceDaily['Last Update']==USLastUpdateDaily

#USAstates=TotalCasesByLastUpdateProvinceDaily[(TotalCasesByLastUpdateProvinceDaily['Country/Region']=='US') & (TotalCasesByLastUpdateProvinceDaily[USLastUpdate_FilterDaily])]

USAstates=TotalCasesByLastUpdateProvinceDaily[USLastUpdate_FilterDaily]

USAstates['Province/State'] = USAstates['Province/State'].astype(str)

#USAstates[['StateName2', 'StateCode2']]=USAstates['Province/State'].str.split(',', expand=True)

#Dictionary of US state codes thank to @rogerallen on github

us_state_abbrev = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Palau': 'PW',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY',

    'Diamond Princess': 'Ship1',

    'Grand Princess': 'Ship2',

}

#Creating data frame for US States Map

#USAstates['StateCode']= np.where(USAstates['StateCode2'].isnull(), USAstates['StateName2'].map(us_state_abbrev), USAstates['StateCode2'])

USAstates['StateCode']=USAstates['Province/State'].map(us_state_abbrev)

inverted_dict = dict( map(reversed, us_state_abbrev.items() ) )

USAstates['StateCode']=USAstates['StateCode'].str.strip()

USAstates['StateName']= USAstates['StateCode'].map(inverted_dict)

USAstatesGroup=USAstates.groupby(['StateName', 'StateCode'],sort=False).agg({"Confirmed": sum, 'Deaths': sum, 'Recovered': sum, 'Active': sum})

USAstatesGroup=USAstatesGroup.reset_index()

fig = px.choropleth(USAstatesGroup, locations="StateCode", 

                    locationmode='USA-states', color="Confirmed",scope='usa', 

                    hover_name="StateName", range_color=[1,10000], 

                    color_continuous_scale="orrd", 

                    title='US States with Confirmed Cases')

fig.update_layout(

    paper_bgcolor='rgba(233,233,233,233)', 

    plot_bgcolor='rgba(0,0,0,0)', 

    coloraxis_showscale=False,

    title_x=0.5,

    margin={"r":0,"t":50,"l":0,"b":20})

fig.update_geos(resolution=110)

fig.show()
map1 = folium.Map(location=[30, 20], tiles = "CartoDB dark_matter", zoom_start=2.5)

TotalCases_LastUpdate['Confirmed']= TotalCases_LastUpdate['Confirmed'].astype(float)

TotalCases_LastUpdate['Deaths']= TotalCases_LastUpdate['Deaths'].astype(float)

TotalCases_LastUpdate['Recovered']= TotalCases_LastUpdate['Recovered'].astype(float)

for i in range(0,len(TotalCases_LastUpdate)):

   folium.Circle(

      location=[TotalCases_LastUpdate.iloc[i]['Lat'], TotalCases_LastUpdate.iloc[i]['Long']],

      popup=

              TotalCases_LastUpdate.iloc[i]['Province/State']+ 

       " Total Confirmed: "+str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Confirmed']))+

       " Total Deaths: "+ str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Deaths']))+

       " Total Recovered: "+ str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Recovered']))+

       " Total Existing: "+ str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Existing'])),

      radius=(math.sqrt(TotalCases_LastUpdate.iloc[i]['Confirmed'])*1200+2 ),

      tooltip = '<li><bold>Country : '+str(TotalCases_LastUpdate.iloc[i]['Country/Region'])+

                '<li><bold>Province : '+str(TotalCases_LastUpdate.iloc[i]['Province/State'])+

                '<li><bold>Confirmed : '+str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Confirmed']))+

                '<li><bold>Deaths : '+str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Deaths']))+

                '<li><bold>Recovered : '+str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Recovered']))+

                '<li><bold>Existing : '+str("{:,.0f}".format(TotalCases_LastUpdate.iloc[i]['Existing'])),

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(map1)

folium.TileLayer('openstreetmap').add_to(map1)

folium.TileLayer('Stamen Terrain').add_to(map1)

folium.TileLayer('Stamen Toner').add_to(map1)

folium.TileLayer('stamenwatercolor').add_to(map1)

folium.TileLayer('cartodbpositron').add_to(map1)

folium.LayerControl().add_to(map1)



map1