import pandas as pd

import plotly 

import plotly.express as px

import plotly.graph_objects as go

import numpy as np



import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')

%matplotlib inline
import cufflinks as cf

import plotly.offline as pyo

from plotly.offline import init_notebook_mode , plot , iplot

import folium
pyo.init_notebook_mode(connected=True)

cf.go_offline()
df = pd.read_excel("../input/corona-dataset/dataset/Covid cases in India.xlsx")

df.head()
df.drop(['S. No.'],axis=1, inplace = True)
df.head(10)
#calculating the total combined cases

df['total cases']= df['Total Confirmed cases (Indian National)']+ df['Total Confirmed cases ( Foreign National )']
df['total cases'].sum()
df
df['Active cases']= df['total cases']-(df['Cured']+df['Death'])
df
df.style.background_gradient(cmap='Blues')
total_cases= df.groupby('Name of State / UT')['Active cases'].sum().sort_values(ascending=False).to_frame()
total_cases.style.background_gradient(cmap='Reds')
#### GRAPHICAL REPRESENTATION OF DATA ####
df.plot(kind='bar',x = 'Name of State / UT',y='Active cases')
plt.bar(df['Name of State / UT'],df['Active cases'])

plt.show()
df.iplot(kind='bar',x = 'Name of State / UT',y='total cases')
##df.plot(kind='scatter',x ='Name of State / UT',y='total cases')

df.iplot(kind='scatter',x = 'Name of State / UT',y='Active cases', mode= 'markers+lines',title='Stat graph', xTitle='States'

         ,yTitle='cases', colors='red', size= 9)
# obejct oriented version of matplotlib and plotly
fig= plt.figure(dpi=200)

axes = fig.add_axes([0,0,5,3])

axes.bar(df['Name of State / UT'],df['Active cases'])

# 4 parmeters are passed in the function the first one is left bottom last two represent height of each axis

axes.set_title("OO GRAPH",size=50)

axes.set_xlabel("States")

axes.set_ylabel("DATA")
#plotly oo

fig= go.Figure()

fig.add_trace(go.Bar(x=df['Name of State / UT'],y=df['Active cases']))

fig.update_layout(title='Cases',xaxis=dict(title='states'),yaxis=dict(title='No. of cases'))
# cordinates of states 
india_cord = pd.read_excel("../input/corona-dataset/dataset/Indian Coordinates.xlsx")
india_cord
df_full = pd.merge(india_cord,df,on='Name of State / UT')

df_full
map= folium.Map(location=[20,70],zoom_start=4,tiles='Stamenterrain')



for lat,long,value,name in zip(df_full['Latitude'],df_full['Longitude'],df_full['total cases'],df_full['Name of State / UT']):

    folium.CircleMarker([lat,long],radius=value*0.4,popup=('<strong>State</strong>: '+str(name).capitalize()+'<br>''<strong>Total Cases</strong>:'+str(value)+'<br>'),color='red',fill_color='red',fill_opacity=0.3).add_to(map)

            

    
map
#checking globally
df_india = pd.read_excel("../input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates= True,sheet_name='India')

df_italy = pd.read_excel("../input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates= True,sheet_name='Italy')

df_korea = pd.read_excel("../input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates= True,sheet_name='Korea')

df_wuhan = pd.read_excel("../input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates= True,sheet_name='Wuhan')
#matplotlib 

fig = plt.figure(dpi=200,figsize=(10,5))

axes= fig.add_axes([0.1, 0.1, 0.8,0.8])

axes.bar(df_india['Date'],df_india['Total Cases'])

axes.set_xlabel("Date ")

axes.set_ylabel('CASES')

axes.set_title("INDIA TIMELINE")
df_india.iplot(kind='bar',x = 'Date',y='Total Cases',colors='gold')
fig = px.bar(df_india,x='Date', y='Total Cases',color='Total Cases', title= 'confirm cases india')

fig.show()
from plotly.subplots import make_subplots
#subplots



fig= make_subplots(rows= 2,cols=2, 

                    specs=[[{'secondary_y': True},{'secondary_y': True}],[{'secondary_y': True},{'secondary_y': True}]],

                    subplot_titles=("S.Korea","Italy","India","Wuhan")

                   )

fig.add_trace(go.Bar(x=df_korea['Date'],y=df_korea['Total Cases'],

                    marker=dict(color=df_korea['Total Cases'],coloraxis='coloraxis')),1,1)



fig.add_trace(go.Bar(x=df_italy['Date'],y=df_italy['Total Cases'],

                    marker=dict(color=df_italy['Total Cases'],coloraxis='coloraxis')),1,2)



fig.add_trace(go.Bar(x=df_india['Date'],y=df_india['Total Cases'],

                    marker=dict(color=df_india['Total Cases'],coloraxis='coloraxis')),2,1)



fig.add_trace(go.Bar(x=df_wuhan['Date'],y=df_wuhan['Total Cases'],

                    marker=dict(color=df_wuhan['Total Cases'],coloraxis='coloraxis')),2,2)
#world map
df= pd.read_csv("../input/corona-dataset/dataset/covid_19_data.csv" ,parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date','Country/Region' : 'Country'},inplace=True)
df.query('Country=="India"')
df.groupby('Date').sum()
confirmed=df.groupby('Date').sum()['Confirmed'].reset_index()

death=df.groupby('Date').sum()['Deaths'].reset_index()

rec =df.groupby('Date').sum()['Recovered'].reset_index()
fig= go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'],y=confirmed['Confirmed'],mode= 'lines+markers',name='confirmed',line=dict(color='blue',width=2)))

fig.add_trace(go.Scatter(x=death['Date'],y=death['Deaths'],mode= 'lines+markers',name='death',line=dict(color='red',width=2))) 

fig.add_trace(go.Scatter(x=rec['Date'],y=rec['Recovered'],mode= 'lines+markers',name='rec',line=dict(color='green',width=2))) 
df_confirmed=pd.read_csv("../input/corona-dataset/dataset/time_series_covid_19_confirmed.csv")
df_confirmed.rename(columns={'Country/Region':'Country'},inplace=True)
df_latlong=pd.merge(df,df_confirmed,on=['Country','Province/State'])
df_latlong
fig= px.density_mapbox(df_latlong,lat='Lat',lon='Long',hover_name="Province/State",hover_data=['Confirmed','Deaths','Recovered'],animation_frame='Date',color_continuous_scale='Portland',radius=7,zoom=0,height=700)

fig.update_layout(title='Worldwide Corona spread')

fig.update_layout(mapbox_style='open-street-map',mapbox_center_lon=0)

fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})