import sys

!conda install --yes --prefix {sys.prefix} -c plotly plotly-orca 
!pip install wget

!pip install calmap

!pip install psutil requests
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from datetime import timedelta

from IPython.display import FileLink

from plotly.subplots import make_subplots

from plotly.offline import plot, iplot, init_notebook_mode

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

import warnings

import calmap

import folium

import wget

import math

import os







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)



if not os.path.exists("images"):

    os.mkdir("images")



if not os.path.exists("Maps"):

    os.mkdir("Maps")

    

if not os.path.exists("Datasets"):

    os.mkdir("Datasets")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Color pallete

Active, Recovered, Confirmed, Deceased, Color_1, Color_2 =  '#ff073a', '#28a745', '#007bff', '#6c757d', '#FE9801', '#FF0F80'
# remove existing files

! rm *.csv



# urls of the files

urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 

        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',

        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']



# download files

for url in urls:

    filename = wget.download(url)
confirmed_df = pd.read_csv('time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('time_series_covid19_recovered_global.csv')

cov = pd.read_csv("../input/covid19-useful-features-by-country/Countries_usefulFeatures.csv")

country_geo = "../input/world-countries/world-countries.json"
print(confirmed_df.shape)

print(deaths_df.shape)

print(recovered_df.shape)
confirmed_df.head()
deaths_df.head()
recovered_df.head()
dates = confirmed_df.columns[4:]



confirmed_df_long = confirmed_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Confirmed')

deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Deaths')

recovered_df_long = recovered_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')



print(confirmed_df_long.shape)

print(deaths_df_long.shape)

print(recovered_df_long.shape)
full_table = pd.merge(left=confirmed_df_long, right=deaths_df_long, how='left',

                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])

full_table = pd.merge(left=full_table, right=recovered_df_long, how='outer',

                      on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])



full_table.head()
full_table.shape
full_table.isna().sum()
full_table['Recovered'] = full_table['Recovered'].fillna(0)

full_table['Recovered'] = full_table['Recovered'].astype('int')

full_table['Deaths'] = full_table['Deaths'].fillna(0)

full_table['Deaths'] = full_table['Deaths'].astype('int')

full_table['Confirmed'] = full_table['Confirmed'].fillna(0)

full_table['Confirmed'] = full_table['Confirmed'].astype('int')

full_table.isna().sum()
full_table['Country/Region'].unique()
full_table['Province/State'].unique()
full_table['Country/Region'] = full_table['Country/Region'].replace('Korea, South', 'South Korea')

full_table['Country/Region'].unique()
full_table = full_table[full_table['Province/State'].str.contains('Recovered')!=True]

full_table = full_table[full_table['Province/State'].str.contains('Grand Princess')!=True]

full_table = full_table[full_table['Province/State'].str.contains('Diamond Princess')!=True]

full_table['Province/State'].unique()
# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

full_table.sample(10)
full_table['Date'] = pd.to_datetime(full_table.Date)

full_table.sort_values(by=['Date'], inplace=True)

full_table.head()
full_table.to_csv('./Datasets/covid_19_cleaned.csv', index=False)
# Grouped by day, country

# =======================



full_grouped = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()



# new cases ======================================================

temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



# renaming columns

temp.columns = ['Country/Region', 'Date', 'No_Of_New_Cases', 'No_Of_New_Deaths', 'No_Of_New_Recovered']

# =================================================================



# merging new values

full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])



# filling na with 0

full_grouped = full_grouped.fillna(0)



# fixing data types

cols = ['No_Of_New_Cases', 'No_Of_New_Deaths', 'No_Of_New_Recovered']

full_grouped[cols] = full_grouped[cols].astype('int')



full_grouped['No_Of_New_Cases'] = full_grouped['No_Of_New_Cases'].apply(lambda x: 0 if x<0 else x)

full_grouped['No_Of_New_Deaths'] = full_grouped['No_Of_New_Deaths'].apply(lambda x: 0 if x<0 else x)

full_grouped['No_Of_New_Recovered'] = full_grouped['No_Of_New_Recovered'].apply(lambda x: 0 if x<0 else x)





full_grouped.sample(10)
cov.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)

full_grouped = pd.merge(full_grouped,cov[['Latitude','Longtitude','Country/Region']], on='Country/Region')

full_grouped.head()
full_grouped.to_csv('./Datasets/covid_19_country_wise.csv', index=False)
# Per Day

# ========



Per_Day = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'No_Of_New_Cases'].sum().reset_index()



Per_Day['% Deaths'] = round((Per_Day['Deaths']/Per_Day['Confirmed'])*100, 2)

Per_Day['% Recovered'] = round((Per_Day['Recovered']/Per_Day['Confirmed'])*100, 2)

Per_Day['% Deaths Per Recovered'] = round((Per_Day['Deaths']/Per_Day['Recovered'])*100, 2)

Per_Day['% New Cases'] = round((Per_Day['No_Of_New_Cases']/Per_Day['Confirmed'])*100, 2)



# no. of countries

Per_Day['Total Countries'] = full_grouped[full_grouped['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len).values



# fillna by 0

cols = ['% Deaths', '% Recovered', '% Deaths Per Recovered', '% New Cases']

Per_Day[cols] = Per_Day[cols].fillna(0)



Per_Day.head()
Country_Wise_Lastest = full_grouped[full_grouped['Date'] == max(full_grouped['Date'])]

Country_Wise_Lastest.head()
# Top 25 Countries

Countries = full_grouped[full_grouped['Date']==max(full_grouped['Date'])]

Countries = Countries.reset_index(drop=True)



Top_Countries = Countries.sort_values(by=['Confirmed'],ascending=False)

Top_Countries = Top_Countries.iloc[:25,:]

Top_Countries = Top_Countries.reset_index(drop=True)





Top_Countries = pd.merge(Top_Countries, cov, on='Country/Region')



Today = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]

Previous_week = full_grouped[full_grouped['Date']==max(full_grouped['Date'])-timedelta(days=7)].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]

Previous_Month = full_grouped[full_grouped['Date']==max(full_grouped['Date'])-timedelta(days=30)].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]



temp = pd.merge(Previous_week, Previous_Month, on='Country/Region', suffixes=(' last week', ' last month'))

temp = pd.merge(Today, temp, on='Country/Region')

temp['New Last Week'] = temp['Confirmed'] - temp['Confirmed last week'] 

temp['New Last Month'] = temp['Confirmed'] - temp['Confirmed last month']



Top_Countries = pd.merge(Top_Countries, temp[['New Last Week', 'New Last Month', 'Country/Region']], on='Country/Region')

Top_Countries['Population in Millions'] = round(Top_Countries['Population_Size'] / 1000000, 2)

Top_Countries['Cases per Million People'] = round(Top_Countries['Confirmed'] / Top_Countries['Population in Millions'])

Top_Countries['Cases per Million People'] = Top_Countries['Cases per Million People'].astype('int')



Top_Countries.head()
Top_Countries_Daily = pd.merge(Top_Countries[['Country/Region']], full_grouped, on='Country/Region', how='left')

Top_Countries_Daily.head()
Per_Day.to_csv('./Datasets/covid_19_Per_Day.csv', index=False)

Country_Wise_Lastest.to_csv('./Datasets/covid_19_Country_Wise_Lastest.csv', index=False)

Top_Countries.to_csv('./Datasets/covid_19_Top_Countries.csv', index=False)

Top_Countries_Daily.to_csv('./Datasets/covid_19_Top_Countries_Daily.csv', index=False)

cov.to_csv('./Datasets/covid_19_Countries_usefulFeatures.csv', index=False)
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)



melted_temp = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(melted_temp, path=["variable"], values="value", height=250, width=1200,

                 color_discrete_sequence=[Active, Recovered, Deceased])

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/treemap_overview.svg")
fig = px.pie(melted_temp, values="value", height=750, names='variable', title='Covid 19',

                 color_discrete_sequence=[Active, Recovered, Deceased])

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/piechart_overview.svg")
temp = full_grouped.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case', height=600,

             title='Cases over time', color_discrete_sequence = [Recovered, Deceased, Active])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()

fig.write_image("images/area_overview.svg")
fig_1 = px.bar(Per_Day, x="Date", y="Confirmed", color_discrete_sequence = [Confirmed])

fig_2 = px.bar(Per_Day, x="Date", y="Active", color_discrete_sequence = [Active])

fig_3 = px.bar(Per_Day, x="Date", y="Recovered", color_discrete_sequence = [Recovered])

fig_4 = px.bar(Per_Day, x="Date", y="Deaths", color_discrete_sequence = [Deceased])

fig_5 = px.bar(Per_Day, x="Date", y="No_Of_New_Cases", color_discrete_sequence = [Color_1])

fig_6 = px.bar(Per_Day, x="Date", y="Total Countries", color_discrete_sequence = [Color_2])



fig = make_subplots(rows=3, cols=2, shared_xaxes=True, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Active Cases', 'Recovered Cases', 'Deaths reported',

                                   'New Cases', 'Countries Affected'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)

fig.add_trace(fig_5['data'][0], row=3, col=1)

fig.add_trace(fig_6['data'][0], row=3, col=2)



fig.update_layout(height=1200, title="Day Wise")

fig.show()

fig.write_image("images/Per_Day_Bar.svg")
fig_1 = px.bar(Per_Day, x="Date", y="Confirmed", color_discrete_sequence = [Confirmed])

fig_2 = px.bar(Per_Day, x="Date", y="Active", color_discrete_sequence = [Active])

fig_3 = px.bar(Per_Day, x="Date", y="Recovered", color_discrete_sequence = [Recovered])

fig_4 = px.bar(Per_Day, x="Date", y="Deaths", color_discrete_sequence = [Deceased])

fig_5 = px.bar(Per_Day, x="Date", y="No_Of_New_Cases", color_discrete_sequence = [Color_1])

fig_6 = px.bar(Per_Day, x="Date", y="Total Countries", color_discrete_sequence = [Color_2])



fig = make_subplots(rows=3, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Active Cases', 'Recovered Cases', 'Deaths reported',

                                   'New Cases', 'Countries Affected'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)

fig.add_trace(fig_5['data'][0], row=3, col=1)

fig.add_trace(fig_6['data'][0], row=3, col=2)



fig.update_layout(height=1200,yaxis_type="log", yaxis2_type="log" ,yaxis3_type="log" ,yaxis4_type="log" ,yaxis5_type="log" ,yaxis6_type="log"

                  , title="Day Wise Logarithmic")

fig.show()

fig.write_image("images/Per_Day_Logarithmic_Bar.svg")
fig_1 = px.bar(Per_Day, x="Date", y="% Recovered", color_discrete_sequence = [Recovered])

fig_2 = px.bar(Per_Day, x="Date", y="% Deaths", color_discrete_sequence = [Active])

fig_3 = px.bar(Per_Day, x="Date", y="% Deaths Per Recovered", color_discrete_sequence = ["#161617"])

fig_4 = px.bar(Per_Day, x="Date", y="% New Cases", color_discrete_sequence = [Color_1])



fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('% Recovered cases', '% Deaths', '% Deaths Per Recovered', '% New Cases'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)





fig.update_layout(height=800, title="% Day Wise")

fig.show()

fig.write_image("images/%_Per_Day_Bar.svg")
fig_1 = px.bar(Per_Day, x="Date", y="% Recovered", color_discrete_sequence = [Recovered])

fig_2 = px.bar(Per_Day, x="Date", y="% Deaths", color_discrete_sequence = [Active])

fig_3 = px.bar(Per_Day, x="Date", y="% Deaths Per Recovered", color_discrete_sequence = ["#161617"])

fig_4 = px.bar(Per_Day, x="Date", y="% New Cases", color_discrete_sequence = [Color_1])



fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('% Recovered cases', '% Deaths', '% Deaths Per Recovered', '% New Cases'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)



fig.update_layout(height=800,yaxis_type="log", yaxis2_type="log" ,yaxis3_type="log" ,yaxis4_type="log", title="% Day Wise Logarithmic")

fig.show()

fig.write_image("images/%_Per_Day_Logarithmic_Bar.svg")
fig_1 = px.line(Per_Day, x="Date", y="Confirmed", color_discrete_sequence = [Confirmed])

fig_2 = px.line(Per_Day, x="Date", y="Active", color_discrete_sequence = [Active])

fig_3 = px.line(Per_Day, x="Date", y="Recovered", color_discrete_sequence = [Recovered])

fig_4 = px.line(Per_Day, x="Date", y="Deaths", color_discrete_sequence = [Deceased])

fig_5 = px.line(Per_Day, x="Date", y="No_Of_New_Cases", color_discrete_sequence = [Color_1])

fig_6 = px.line(Per_Day, x="Date", y="Total Countries", color_discrete_sequence = [Color_2])



fig = make_subplots(rows=3, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Active Cases', 'Recovered Cases', 'Deaths reported',

                                   'New Cases', 'Countries Affected'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)

fig.add_trace(fig_5['data'][0], row=3, col=1)

fig.add_trace(fig_6['data'][0], row=3, col=2)



fig.update_layout(height=1200, title="Day Wise")

fig.show()

fig.write_image("images/Per_Day_Line.svg")
fig_1 = px.line(Per_Day, x="Date", y="Confirmed", color_discrete_sequence = [Confirmed])

fig_2 = px.line(Per_Day, x="Date", y="Active", color_discrete_sequence = [Active])

fig_3 = px.line(Per_Day, x="Date", y="Recovered", color_discrete_sequence = [Recovered])

fig_4 = px.line(Per_Day, x="Date", y="Deaths", color_discrete_sequence = [Deceased])

fig_5 = px.line(Per_Day, x="Date", y="No_Of_New_Cases", color_discrete_sequence = [Color_1])

fig_6 = px.line(Per_Day, x="Date", y="Total Countries", color_discrete_sequence = [Color_2])



fig = make_subplots(rows=3, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Active Cases', 'Recovered Cases', 'Deaths reported',

                                   'New Cases', 'Countries Affected'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)

fig.add_trace(fig_5['data'][0], row=3, col=1)

fig.add_trace(fig_6['data'][0], row=3, col=2)



fig.update_layout(height=1200,yaxis_type="log", yaxis2_type="log" ,yaxis3_type="log" ,yaxis4_type="log" ,yaxis5_type="log" ,yaxis6_type="log"

                  , title="Day Wise Logarithmic")

fig.show()

fig.write_image("images/Per_Day_Logarithmic_Line.svg")
fig_1 = px.line(Per_Day, x="Date", y="% Recovered", color_discrete_sequence = [Recovered])

fig_2 = px.line(Per_Day, x="Date", y="% Deaths", color_discrete_sequence = [Active])

fig_3 = px.line(Per_Day, x="Date", y="% Deaths Per Recovered", color_discrete_sequence = ["#161617"])

fig_4 = px.line(Per_Day, x="Date", y="% New Cases", color_discrete_sequence = [Color_1])



fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('% Recovered cases', '% Deaths', '% Deaths Per Recovered', '% New Cases'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)





fig.update_layout(height=800, title="% Day Wise")

fig.show()

fig.write_image("images/%_Per_Day_Line.svg")
fig_1 = px.line(Per_Day, x="Date", y="% Recovered", color_discrete_sequence = [Recovered])

fig_2 = px.line(Per_Day, x="Date", y="% Deaths", color_discrete_sequence = [Active])

fig_3 = px.line(Per_Day, x="Date", y="% Deaths Per Recovered", color_discrete_sequence = ["#161617"])

fig_4 = px.line(Per_Day, x="Date", y="% New Cases", color_discrete_sequence = [Color_1])



fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('% Recovered cases', '% Deaths', '% Deaths Per Recovered', '% New Cases'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)



fig.update_layout(height=800,yaxis_type="log", yaxis2_type="log" ,yaxis3_type="log" ,yaxis4_type="log", title="% Day Wise Logarithmic")

fig.show()

fig.write_image("images/%_Per_Day_Logarithmic_Line.svg")
# World wide

temp = full_grouped[full_grouped['Date'] == max(full_grouped['Date'])]



_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1.5)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],

        color=Confirmed, fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Active : '+str(temp.iloc[i]['Active'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths'])+

                    '<li><bold>New Cases : '+str(temp.iloc[i]['No_Of_New_Cases'])+

                    '<li><bold>New Death : '+str(temp.iloc[i]['No_Of_New_Deaths'])+

                    '<li><bold>New Recovered : '+str(temp.iloc[i]['No_Of_New_Recovered']),

        radius=int(temp.iloc[i]['Confirmed'])).add_to(_map)

_map.save('./Maps/Confirmed.html')

_map
_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1.5)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],

        color=Active, fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Active : '+str(temp.iloc[i]['Active'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths'])+

                    '<li><bold>New Cases : '+str(temp.iloc[i]['No_Of_New_Cases'])+

                    '<li><bold>New Death : '+str(temp.iloc[i]['No_Of_New_Deaths'])+

                    '<li><bold>New Recovered : '+str(temp.iloc[i]['No_Of_New_Recovered']),

        radius=int(temp.iloc[i]['Active'])).add_to(_map)



_map.save('./Maps/Active.html')

_map
_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1.5)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],

        color=Recovered, fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Active : '+str(temp.iloc[i]['Active'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(temp.iloc[i]['Recovered'])+

                    '<li><bold>New Cases : '+str(temp.iloc[i]['No_Of_New_Cases'])+

                    '<li><bold>New Death : '+str(temp.iloc[i]['No_Of_New_Deaths'])+

                    '<li><bold>New Recovered : '+str(temp.iloc[i]['No_Of_New_Recovered']),

        radius=int(temp.iloc[i]['Recovered'])**1.05).add_to(_map)



_map.save('./Maps/Recovered.html')

_map
_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1.5)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],

        color=Deceased, fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Active : '+str(temp.iloc[i]['Active'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths'])+

                    '<li><bold>New Cases : '+str(temp.iloc[i]['No_Of_New_Cases'])+

                    '<li><bold>New Death : '+str(temp.iloc[i]['No_Of_New_Deaths'])+

                    '<li><bold>New Recovered : '+str(temp.iloc[i]['No_Of_New_Recovered']),

        radius=int(temp.iloc[i]['Deaths'])**1.2).add_to(_map)



_map.save('./Maps/Deceased.html')

_map
fig = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names', color=np.log(full_grouped["Confirmed"]), 

                    hover_name="Country/Region",

                    title='Confirmed Cases', color_continuous_scale=px.colors.sequential.Blues)

fig.update(layout_coloraxis_showscale=False)

fig.show()

fig.write_image("images/Confirmed_choropleth.svg")
fig = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names', color=np.log(full_grouped["Active"]), 

                    hover_name="Country/Region",

                    title='Active Cases', color_continuous_scale=px.colors.sequential.Reds)

fig.update(layout_coloraxis_showscale=False)

fig.show()

fig.write_image("images/Active_choropleth.svg")
fig = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names', color=np.log(full_grouped["Recovered"]), 

                    hover_name="Country/Region",

                    title='Recovered Cases', color_continuous_scale=px.colors.sequential.Greens)

fig.update(layout_coloraxis_showscale=False)

fig.show()

fig.write_image("images/Recovered_choropleth.svg")
fig = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names', color=np.log(full_grouped["Deaths"]), 

                    hover_name="Country/Region",

                    title='Deceased Cases', color_continuous_scale=px.colors.sequential.Greys)

fig.update(layout_coloraxis_showscale=False)

fig.show()

fig.write_image("images/Deceased_choropleth.svg")
temp = full_grouped[full_grouped['Date'] == max(full_grouped['Date'])]



_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1.5)



_map.choropleth( geo_data=country_geo ,data=cov,

               columns=['Country_Code', 'Mean_Age'],

                key_on='feature.id',

               legend_name="Mean Age",

               fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.5)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],

        color=Active, fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Active : '+str(temp.iloc[i]['Active'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths'])+

                    '<li><bold>New Cases : '+str(temp.iloc[i]['No_Of_New_Cases'])+

                    '<li><bold>New Death : '+str(temp.iloc[i]['No_Of_New_Deaths'])+

                    '<li><bold>New Recovered : '+str(temp.iloc[i]['No_Of_New_Recovered']),

        radius=int(temp.iloc[i]['Confirmed'])).add_to(_map)

_map.save('./Maps/Mean_Age.html')

_map
temp = full_grouped[full_grouped['Date'] == max(full_grouped['Date'])]



_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1.5)



_map.choropleth( geo_data=country_geo ,data=cov,

               columns=['Country_Code', 'Tourism'],

                key_on='feature.id',

               legend_name="Tourism",

               fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.5)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],

        color=Active, fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Active : '+str(temp.iloc[i]['Active'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths'])+

                    '<li><bold>New Cases : '+str(temp.iloc[i]['No_Of_New_Cases'])+

                    '<li><bold>New Death : '+str(temp.iloc[i]['No_Of_New_Deaths'])+

                    '<li><bold>New Recovered : '+str(temp.iloc[i]['No_Of_New_Recovered']),

        radius=int(temp.iloc[i]['Confirmed'])).add_to(_map)

_map.save('./Maps/Tourism.html')

_map
cov['Population in Millions'] = round(cov['Population_Size'] / 1000000, 2)



temp = full_grouped[full_grouped['Date'] == max(full_grouped['Date'])]



_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1.5)



_map.choropleth( geo_data=country_geo ,data=cov,

               columns=['Country_Code', 'Population in Millions'],

                key_on='feature.id',

               legend_name="Population in Millions",

               fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.5)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Latitude'], temp.iloc[i]['Longtitude']],

        color=Active, fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Active : '+str(temp.iloc[i]['Active'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths'])+

                    '<li><bold>New Cases : '+str(temp.iloc[i]['No_Of_New_Cases'])+

                    '<li><bold>New Death : '+str(temp.iloc[i]['No_Of_New_Deaths'])+

                    '<li><bold>New Recovered : '+str(temp.iloc[i]['No_Of_New_Recovered']),

        radius=int(temp.iloc[i]['Confirmed'])).add_to(_map)

_map.save('./Maps/Population.html')

_map
fig = px.treemap(Country_Wise_Lastest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Dark24)

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/Country_Wise_Treemap_Confirmed.svg")
fig = px.treemap(Country_Wise_Lastest.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Deaths", height=700,

                 title='Number Deaths Reported',

                 color_discrete_sequence = px.colors.qualitative.Dark24)

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/Country_Wise_Treemap_Deaths.svg")
fig = px.treemap(Country_Wise_Lastest.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Recovered", height=700,

                 title='Number of Recovered Cases',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/Country_Wise_Treemap_Recovered.svg")
# confirmed - Active

fig_1 = px.bar(Top_Countries.sort_values('Confirmed'), x="Confirmed", y="Country/Region", 

               text='Confirmed', orientation='h', color_discrete_sequence = [Confirmed])

fig_2 = px.bar(Top_Countries.sort_values('Active'), x="Active", y="Country/Region", 

               text='Active', orientation='h', color_discrete_sequence = [Active])



# recovered - Deaths

fig_3 = px.bar(Top_Countries.sort_values('Recovered'), x="Recovered", y="Country/Region", 

               text='Recovered', orientation='h', color_discrete_sequence = [Recovered])

fig_4 = px.bar(Top_Countries.sort_values('Deaths'), x="Deaths", y="Country/Region", 

               text='Deaths', orientation='h', color_discrete_sequence = [Deceased])



# Cases Today - Recovered Today

fig_5 = px.bar(Top_Countries.sort_values('No_Of_New_Cases'), x="No_Of_New_Cases", y="Country/Region", 

               text='No_Of_New_Cases', orientation='h', color_discrete_sequence = [Color_1])

fig_6 = px.bar(Top_Countries.sort_values('No_Of_New_Recovered'), x="No_Of_New_Recovered", y="Country/Region", 

               text='No_Of_New_Recovered', orientation='h', color_discrete_sequence = [Color_2])



# Deaths Today - Cases Per Million People

fig_7 = px.bar(Top_Countries.sort_values('No_Of_New_Deaths'), x="No_Of_New_Deaths", y="Country/Region", 

               text='No_Of_New_Deaths', orientation='h', color_discrete_sequence = ['#222222'])

fig_8 = px.bar(Top_Countries.sort_values('Cases per Million People'), x="Cases per Million People", y="Country/Region", 

               text='Cases per Million People', orientation='h', color_discrete_sequence = ['#5e027d'])



# New Cases Last Week - New Cases Last Month

fig_9 = px.bar(Top_Countries.sort_values('New Last Week'), x="New Last Week", y="Country/Region", 

               text='New Last Week', orientation='h', color_discrete_sequence = ['#9c0338'])

fig_10 = px.bar(Top_Countries.sort_values('New Last Month'), x="New Last Month", y="Country/Region", 

               text='New Last Month', orientation='h', color_discrete_sequence = ['#07a1fa'])



# plot

fig = make_subplots(rows=5, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.03,

                    subplot_titles=('Confirmed Cases', 'Active Cases', 'Recovered', 'Deaths Reported', 

                                    'Cases Today', 'Recovered Today', 'Deaths Reported Today',

                                   'Cases per Million People', 'New Cases Last Week',

                                   'New Cases Last Month'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=2, col=1)

fig.add_trace(fig_4['data'][0], row=2, col=2)

fig.add_trace(fig_5['data'][0], row=3, col=1)

fig.add_trace(fig_6['data'][0], row=3, col=2)

fig.add_trace(fig_7['data'][0], row=4, col=1)

fig.add_trace(fig_8['data'][0], row=4, col=2)

fig.add_trace(fig_9['data'][0], row=5, col=1)

fig.add_trace(fig_10['data'][0], row=5, col=2)



fig.update_layout(height=5500)

fig.show()

fig.write_image("images/Top_Countries_Overview.svg")
fig = px.scatter(Top_Countries.sort_values('Deaths', ascending=False), 

                 x='Confirmed', y='Deaths', color='Country/Region', size='Confirmed', height=700,

                 text='Country/Region', log_x=True, log_y=True, title='Deaths vs Confirmed (Logarithmic Scale)')

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout(height=1500)

fig.show()

fig.write_image("images/Top_Countries_Scatter_DvC.svg")
fig = px.scatter(Top_Countries.sort_values('Recovered', ascending=False), 

                 x='Confirmed', y='Recovered', color='Country/Region', size='Confirmed', height=700,

                 text='Country/Region', log_x=True, log_y=True, title='Recovered vs Confirmed (Logarithmic Scale)')

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout(height=1500)

fig.show()

fig.write_image("images/Top_Countries_Scatter_RvC.svg")
fig = go.Figure(data=[

    go.Bar(name='Active', x=Top_Countries['Country/Region'].head(10), y=Top_Countries['Active'], marker_color=Confirmed),

    go.Bar(name='Recovered', x=Top_Countries['Country/Region'].head(10), y=Top_Countries['Recovered'], marker_color=Active),

    go.Bar(name='Deaths', x=Top_Countries['Country/Region'].head(10), y=Top_Countries['Deaths'], marker_color=Deceased)

           ])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()

fig.write_image("images/Top_Countries_GroupedBar.svg")
fig = go.Figure(data=[

    go.Bar(name='Active', x=Top_Countries['Country/Region'].head(20), y=Top_Countries['Active'], marker_color=Confirmed),

    go.Bar(name='Recovered', x=Top_Countries['Country/Region'].head(20), y=Top_Countries['Recovered'], marker_color=Active),

    go.Bar(name='Deaths', x=Top_Countries['Country/Region'].head(20), y=Top_Countries['Deaths'], marker_color=Deceased)

           ])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.show()

fig.write_image("images/Top_Countries_StackedBar.svg")
fig = px.treemap(Top_Countries.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Dark24)

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/Top_Country_Treemap_Confirmed.svg")
fig = px.treemap(Top_Countries.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Deaths", height=700,

                 title='Number of Deaths Reported',

                 color_discrete_sequence = px.colors.qualitative.Dark24)

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/Top_Country_Treemap_Deaths.svg")
fig = px.treemap(Top_Countries.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Recovered", height=700,

                 title='Number of Recovered Cases',

                 color_discrete_sequence = px.colors.qualitative.Dark24)

fig.data[0].textinfo = 'label+text+value'

fig.show()

fig.write_image("images/Top_Country_Treemap_Recovered.svg")
fig = px.bar(Top_Countries_Daily, x="Date", y="Confirmed", color='Country/Region', height=750,

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.Edge)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedBar_Confirmed.svg")
fig = px.bar(Top_Countries_Daily, x="Date", y="Deaths", color='Country/Region', height=750,

             title='Deaths', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedBar_Deaths.svg")
fig = px.bar(Top_Countries_Daily, x="Date", y="No_Of_New_Cases", color='Country/Region', height=750,

             title='New cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedBar_New.svg")
fig = px.bar(Top_Countries_Daily, x="Date", y="Recovered", color='Country/Region', height=750,

             title='Recovered', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedBar_Recovered.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="Confirmed", color='Country/Region', height=750,

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.Edge)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_Confirmed.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="Deaths", color='Country/Region', height=750,

             title='Deaths', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_Deaths.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="No_Of_New_Cases", color='Country/Region', height=750,

             title='New cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_New.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="Recovered", color='Country/Region', height=750,

             title='Recovered', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_Recovered.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="Confirmed", color='Country/Region', height=750,

             title='Confirmed (Logarithmic)', color_discrete_sequence = px.colors.cyclical.Edge)

fig.update_layout(yaxis_type='log')

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_Confirmed_Log.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="Deaths", color='Country/Region', height=750,

             title='Deaths (Logarithmic)', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.update_layout(yaxis_type='log')

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_Deaths_Log.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="No_Of_New_Cases", color='Country/Region', height=750,

             title='New Cases (Logarithmic)', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.update_layout(yaxis_type='log')

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_New_Log.svg")
fig = px.line(Top_Countries_Daily, x="Date", y="Recovered", color='Country/Region', height=750,

             title='Recovered (Logarithmic)', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.update_layout(yaxis_type='log')

fig.show()

fig.write_image("images/Top_Country_Daily_GroupedLine_Recovered_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Confirmed'], name=Country), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Cases in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_Confirmed.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Active'], name=Country), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Active Cases in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_Active.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Deaths'], name=Country), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Fatalities Reported in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_Deaths.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'No_Of_New_Cases'], name=Country), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of New Cases Reported in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_New.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Confirmed']), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Cases in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_Confirmed_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Active'], name=Country), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Active Cases in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_Active_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Deaths'], name=Country), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Fatalities Reported in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_Deaths_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Bar(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'No_Of_New_Cases'], name=Country), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of New Cases Reported in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Bar_New_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Confirmed']), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Cases in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_Confirmed.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Active'], name=Country), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Active Cases in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_Active.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Deaths'], name=Country), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Fatalities Reported in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_Deaths.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'No_Of_New_Cases'], name=Country), row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of New Cases Reported in each Country")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_New.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Confirmed']), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Cases in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_Confirmed_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Active'], name=Country), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Active Cases in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_Active_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'Deaths'], name=Country), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of Fatalities Reported in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_Deaths_Log.svg")
Countries = Top_Countries_Daily['Country/Region'].unique()



n_cols = 3

n_rows = math.ceil(len(Countries)/n_cols)



fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=Countries)



for idx, Country in enumerate(Countries):

    row = int((idx/n_cols)+1)

    col = int((idx%n_cols)+1)

    fig.add_trace(go.Line(x=Top_Countries_Daily['Date'], y=Top_Countries_Daily.loc[Top_Countries_Daily['Country/Region']==Country, 'No_Of_New_Cases'], name=Country), row=row, col=col)

    fig.update_yaxes(type='log', row=row, col=col)

    

fig.update_layout(height=2500, title_text="No. of New Cases Reported in each Country (Logarithmic)")    

fig.show()

fig.write_image("images/Top_Country_Daily_Seperate_Line_New_Log.svg")
# Download All files

!pwd

! rm *.zip

!zip -r /kaggle/working/Files.zip /kaggle/working

FileLink(r'Files.zip')