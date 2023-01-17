# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Ploting and visualisations 
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px 
from plotly.offline import download_plotlyjs,init_notebook_mode, iplot
import plotly.tools as tls 
import plotly.figure_factory as ff 
py.init_notebook_mode(connected=True)
# ----------------------- #


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
confirmed_case = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
confirmed_case_us = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv')
death_case = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv')
covid_19_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
covid_19_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_19_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
COVID19_line_list_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
COVID19_open_line_list = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
display(covid_19_data.info(),covid_19_data.head())
## checking if any null values in the dataset
covid_19_data.isnull().sum()
def missing_value(covid_19_data):
    null_data = pd.DataFrame(((covid_19_data.isnull().sum())/len(covid_19_data['Confirmed']))*100,columns = ['percentage'])
    null_data = null_data.round(2)
    trace = go.Bar(x=null_data.index,y=null_data['percentage'],opacity=0.5,text = null_data['percentage'],textposition = 'auto', marker=dict(color = 'turquoise',line=dict(color= 'green',width=1.5)))
    
    layout = dict(title='Percentage of null data in dataset')
    
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)
missing_value(covid_19_data)
covid_19_data['Province/State'] = covid_19_data['Province/State'].fillna('Unknown')
display(covid_19_data.isnull().sum(),covid_19_data.head())
covid_19_data[["Confirmed","Deaths","Recovered"]] = covid_19_data[["Confirmed","Deaths","Recovered"]].astype(int)
covid_19_data['Country/Region'] = covid_19_data['Country/Region'].replace('Mainland China','China')
covid_19_data['Active'] = covid_19_data['Confirmed'] - (covid_19_data['Deaths']+covid_19_data['Recovered']) 
# Used just to see all the rows for critical analysis. 
# pd.set_option("display.max.rows", None)
## covid_19_data[covid_19_data['ObservationDate'] == covid_19_data['ObservationDate'].max()] to find the latest data on dases of the day 
## We then sum the latest day data and group them by 'Countries'

df1 = covid_19_data[covid_19_data['ObservationDate'] == covid_19_data['ObservationDate'].max()].groupby(["Country/Region"])[["Confirmed","Active","Recovered","Deaths"]].sum().reset_index()
## We are adding the latitude and longitude with df1 table.
df2 = confirmed_case[['Country/Region','Lat','Long']].reset_index()

## We had multiple loction for the same country so we removed the duplicate values. 
df2 = df2.drop_duplicates(subset = ["Country/Region"])
# We merged the two dataframes to have location of the countries also .. 
merge_1_2 = pd.merge(df1,df2, on=['Country/Region'], how='inner')

## Droping Column index as not required
merge_1_2 = merge_1_2.drop(columns=['index'])

display(merge_1_2,merge_1_2.info())
## Our tool for ploting these beautiful maps..
import folium
## We need coordinates to figure out where we can place our pointers 
locations = merge_1_2[['Lat', 'Long']]
locationlist = locations.values.tolist()
from folium.plugins import MarkerCluster
map2 = folium.Map(location=[20.5937, 0], tiles='CartoDB dark_matter', zoom_start=2)

marker_cluster = MarkerCluster().add_to(map2)

for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=merge_1_2['Country/Region'][point]).add_to(marker_cluster)
map2
map = folium.Map(location= [20.5937, 0],tiles='CartoDB dark_matter', zoom_start=2)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=merge_1_2['Country/Region'][point]).add_to(map)
map
# Make an empty map
m = folium.Map(location=[20,0], tiles="CartoDB dark_matter", zoom_start=2)
 
# I can add marker one by one on the map
for i in range(0,len(merge_1_2)):
    folium.Circle(
      location=locationlist[i],
      popup = (
        "<strong>Country/Region:</strong> {Country}</br>"
        "<strong>Confirmed case:</strong> {Confirmed}<br>"
    ).format(Country=str(merge_1_2.iloc[i]['Country/Region']), Confirmed=str(merge_1_2.iloc[i]['Confirmed'])),
      radius=merge_1_2.iloc[i]['Confirmed']/2.5,
      color='darkorange',
      fill=True,
      fill_color='darkorange'
   ).add_to(m)
 
# Save it as html
#m.save('mymap.html')
m
def bar_plot(merge_1_2,var):
    Countries_data = merge_1_2.nlargest(10,[var])
    #Countries_data['per'] = (Countries_data[var]/sum(Countries_data[var]))*100
    trace = go.Bar(y=Countries_data[var],x=Countries_data['Country/Region'],opacity=0.5,
                   text = Countries_data[var],textposition = 'auto',
                   marker_color =  px.colors.qualitative.Bold
                   )
    
    layout = dict(title='Top 10 countries with highest {} cases'.format(var))
    
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)
bar_plot(merge_1_2,'Confirmed')
# Make an empty map
m = folium.Map(location=[20,0], tiles="CartoDB dark_matter", zoom_start=2)
 
# I can add marker one by one on the map
for i in range(0,len(merge_1_2)):
    folium.Circle(
      location=locationlist[i],
      popup = (
        "<strong>Country/Region:</strong> {Country}</br>"
        "<strong>Active case:</strong> {Active}<br>"
    ).format(Country=str(merge_1_2.iloc[i]['Country/Region']), Active=str(merge_1_2.iloc[i]['Active'])),
      radius=merge_1_2.iloc[i]['Active']/2.5,
      color='#9ACD32',
      fill=True,
      fill_color='#9ACD32'
   ).add_to(m)
 
# Save it as html
#m.save('mymap.html')
m
bar_plot(merge_1_2,'Active')
# Make an empty map
m = folium.Map(location=[20,0], tiles="CartoDB dark_matter", zoom_start=2)
 
# I can add marker one by one on the map
for i in range(0,len(merge_1_2)):
    folium.Circle(
      location=locationlist[i],
      popup = (
        "<strong>Country/Region:</strong> {Country}</br>"
        "<strong>Death case:</strong> {Deaths}<br>"
    ).format(Country=str(merge_1_2.iloc[i]['Country/Region']), Deaths=str(merge_1_2.iloc[i]['Deaths'])),
      radius=merge_1_2.iloc[i]['Deaths']/.125,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
 
# Save it as html
#m.save('mymap.html')
m
bar_plot(merge_1_2,'Deaths')
# Make an empty map
m = folium.Map(location=[20,0], tiles="CartoDB dark_matter", zoom_start=2)
 
# I can add marker one by one on the map
for i in range(0,len(merge_1_2)):
    folium.Circle(
      location=locationlist[i],
      popup = (
        "<strong>Country/Region:</strong> {Country}</br>"
        "<strong>Recovered patient:</strong> {Recovered}<br>"
    ).format(Country=str(merge_1_2.iloc[i]['Country/Region']), Recovered=str(merge_1_2.iloc[i]['Recovered'])),
      radius=merge_1_2.iloc[i]['Recovered']/1,
      color='#028ACA',
      fill=True,
      fill_color='#028ACA'
   ).add_to(m)
 
# Save it as html
#m.save('mymap.html')
m
bar_plot(merge_1_2,'Recovered')
