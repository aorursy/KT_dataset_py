import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
#style.use('ggplot')
%matplotlib inline

import plotly
import plotly.express as px
import plotly.graph_objects as go
plt.rcParams['figure.figsize']=13,10
import cufflinks as cf
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot

import folium
pyo.init_notebook_mode(connected=True)
cf.go_offline()
df=pd.read_excel("/kaggle/input/dataset/Covid cases in India.xlsx")
df
df.drop(['S. No.'],axis=1,inplace=True)
df['Total Cases']=df['Total Confirmed cases (Indian National)']+df['Total Confirmed cases ( Foreign National )']
df
total_cases=df['Total Cases'].sum()
print('The total number of cases till now in India is ',total_cases)
df['Active Cases']=df['Total Cases']-(df['Death']+df['Cured'])
df
df.style.background_gradient(cmap='Reds')
Total_Active_Cases=df.groupby('Name of State / UT')['Active Cases'].sum().sort_values(ascending=False).to_frame()
Total_Active_Cases.style.background_gradient(cmap='Reds')
#Pandas vis
df.plot(kind='bar',x='Name of State / UT',y='Total Cases')
#Plotly
df.iplot(kind='bar',x='Name of State / UT',y='Total Cases')
#Matplotlib vis

plt.bar(df['Name of State / UT'],df['Total Cases'])
px.bar(df,x='Name of State / UT',y='Total Cases')
df.plot(kind='scatter',x='Name of State / UT',y='Total Cases')
plt.scatter(df['Name of State / UT'],df['Total Cases'])
df.iplot(kind='scatter',
         x='Name of State / UT',
         y='Total Cases',
         mode='markers+lines',
         title='My Graph',
         xTitle='Name of State / UT',
         yTitle='Total Cases',
         colors='red',size=20)
px.scatter(df,x='Name of State / UT',y='Total Cases')
#Matplotlib
fig=plt.figure(figsize=(20,10),dpi=200)
axes=fig.add_axes([0,0,1,1])
axes.bar(df['Name of State / UT'],df['Total Cases'])
axes.set_title("Total Cases in India")
axes.set_xlabel("Name of State / UT")
axes.set_ylabel("Total Cases")
plt.show()

#plotly
fig=go.Figure()
fig.add_trace(go.Bar(x=df['Name of State / UT'],y=df['Total Cases']))
fig.update_layout(title='Total Cases in India',xaxis=dict(title='Name of State / UT'),yaxis=dict(title='Total Cases'))
indian_cord=pd.read_excel("/kaggle/input/dataset/Indian Coordinates.xlsx")
indian_cord
new_df = pd.merge(indian_cord,df, on='Name of State / UT')
new_df
map=folium.Map(location=[20,70],zoom_start=4,tiles='Stamenterrain')

for lat,long,value, name in zip(new_df['Latitude'],new_df['Longitude'],new_df['Total Cases'],new_df['Name of State / UT']):
    folium.CircleMarker([lat,long],radius=value*0.8,popup=('<strong>State</strong>: '
                                                           +str(name).capitalize()+'<br>''<strong>Total Cases</strong>: ' 
                                                           + str(value)+ '<br>'),color='red',fill_color='red',fill_opacity=0.3).add_to(map)

map
df_india=pd.read_excel("/kaggle/input/dataset/per_day_cases.xlsx",parse_dates=True,sheet_name="India")
df_italy=pd.read_excel("/kaggle/input/dataset/per_day_cases.xlsx",parse_dates=True,sheet_name="Italy")
df_korea=pd.read_excel("/kaggle/input/dataset/per_day_cases.xlsx",parse_dates=True,sheet_name="Korea")
df_wuhan=pd.read_excel("/kaggle/input/dataset/per_day_cases.xlsx",parse_dates=True,sheet_name="Wuhan")
df_india.head()
#Matplotlib
fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(df_india["Date"],df_india["Total Cases"],color='blue')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("Confirmed cases in India")
plt.show()
#plotly

df_india.iplot(kind='bar' , x="Date",y="Total Cases",title='Confirmed cases in India')
#plotly Express

fig=px.bar(df_india,x="Date",y="Total Cases",title='Confirmed cases in India')
fig.show()
df_italy.head()
df_korea.head()
df_wuhan.head()
#Matplotlib
fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(df_italy["Date"],df_italy["Total Cases"],color='red')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("Confirmed cases in italy")
plt.show()

#Matplotlib
fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(df_korea["Date"],df_korea["Total Cases"],color='orange')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("Confirmed cases in korea")
plt.show()

#Matplotlib
fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(df_wuhan["Date"],df_wuhan["Total Cases"],color='black')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("Confirmed cases in wuhan")
plt.show()
#plotly
df_italy.iplot(kind='bar' , x="Date",y="Total Cases",color = 'red',title='Confirmed cases in italy')

#plotly
df_korea.iplot(kind='bar' , x="Date",y="Total Cases",color = 'green',title='Confirmed cases in korea')

#plotly
df_wuhan.iplot(kind='bar' , x="Date",y="Total Cases",color = 'purple',title='Confirmed cases in wuhan')
#plotly express
fig=px.bar(df_italy,x="Date",y="Total Cases",title='Confirmed cases in Italy')
fig.show()


#plotly express
fig=px.bar(df_korea,x="Date",y="Total Cases",title='Confirmed cases in Korea')
fig.show()


#plotly express
fig=px.bar(df_wuhan,x="Date",y="Total Cases",title='Confirmed cases in Wuhan')
fig.show()
#Matplotlib
fig=plt.figure(figsize=(20,8))
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.scatter(df_india["Date"],df_india["Total Cases"],color='blue',marker='+')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("Confirmed cases in India")
plt.show()

#Plotly
df_india.iplot(kind='scatter',x='Date',y='Total Cases',mode='lines+markers')

#plotly Express

fig=px.scatter(df_india,x="Date",y="Total Cases",color='Total Cases',title='Confirmed cases in India')
fig.show()
from plotly.subplots import make_subplots
fig=make_subplots(
    rows=2,cols=2,
    specs=[[{"secondary_y":True},{"secondary_y":True}],[{"secondary_y":True},{"secondary_y":True}]],
    subplot_titles=("S.Korea","Italy","India","Wuhan"))
fig.add_trace(go.Bar(x=df_korea['Date'],y=df_korea['Total Cases'],
                    marker=dict(color=df_korea['Total Cases'],coloraxis="coloraxis")),1,1)

fig.add_trace(go.Bar(x=df_italy['Date'],y=df_italy['Total Cases'],
                    marker=dict(color=df_italy['Total Cases'],coloraxis="coloraxis")),1,2)

fig.add_trace(go.Bar(x=df_india['Date'],y=df_india['Total Cases'],
                    marker=dict(color=df_india['Total Cases'],coloraxis="coloraxis")),2,1)

fig.add_trace(go.Bar(x=df_wuhan['Date'],y=df_wuhan['Total Cases'],
                    marker=dict(color=df_wuhan['Total Cases'],coloraxis="coloraxis")),2,2)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'),showlegend=False,title_text="Total Cases in 4 Countries")

#fig.update_layout(plot_bgcolor='rgb(230,230,230)')
fig=make_subplots(
    rows=2,cols=2,
    specs=[[{"secondary_y":True},{"secondary_y":True}],[{"secondary_y":True},{"secondary_y":True}]],
    subplot_titles=("S.Korea","Italy","India","Wuhan"))
fig.add_trace(go.Scatter(x=df_korea['Date'],y=df_korea['Total Cases'],
                    marker=dict(color=df_korea['Total Cases'],coloraxis="coloraxis")),1,1)

fig.add_trace(go.Scatter(x=df_italy['Date'],y=df_italy['Total Cases'],
                    marker=dict(color=df_italy['Total Cases'],coloraxis="coloraxis")),1,2)

fig.add_trace(go.Scatter(x=df_india['Date'],y=df_india['Total Cases'],
                    marker=dict(color=df_india['Total Cases'],coloraxis="coloraxis")),2,1)

fig.add_trace(go.Scatter(x=df_wuhan['Date'],y=df_wuhan['Total Cases'],
                    marker=dict(color=df_wuhan['Total Cases'],coloraxis="coloraxis")),2,2)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'),showlegend=False,title_text="Total Cases in 4 Countries")

fig.update_layout(plot_bgcolor='rgb(230,230,230)')
df=pd.read_csv('/kaggle/input/dataset/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date','Country/Region':'Country'},inplace=True)
df
df.info()
#df.Country.str.split(expand=True).stack().value_counts()
name=[]
for i in df['Country']:
     if  i == i and i not in name: 
                name.append(i)
print(*name,sep='\n')

name=[]
for i in df['Province/State']:
     if  i == i and i not in name: 
                name.append(i)
print(*name,sep='\n')

df.groupby('Date').sum()
confirmed=df.groupby('Date').sum()['Confirmed'].reset_index()
death=df.groupby('Date').sum()['Deaths'].reset_index()
rec=df.groupby('Date').sum()['Recovered'].reset_index()
fig=go.Figure()
fig.add_trace(go.Scatter(x=confirmed['Date'],y=confirmed['Confirmed'],mode='lines+markers',name='Confirmed',line=dict(color='blue',width=2)))
fig.add_trace(go.Scatter(x=death['Date'],y=death['Deaths'],mode='lines+markers',name='Deaths',line=dict(color='red',width=2)))
fig.add_trace(go.Scatter(x=rec['Date'],y=rec['Recovered'],mode='lines+markers',name='Recovered',line=dict(color='green',width=2)))
df_confirmed=pd.read_csv('/kaggle/input/dataset/time_series_covid_19_confirmed.csv')
df_confirmed.rename(columns={'Country/Region':'Country'},inplace=True)
df
df_main1=pd.merge(df,df_confirmed,on=['Country','Province/State'])
df_main1
fig=px.density_mapbox(df_main1,lat="Lat",lon="Long",hover_name="Province/State",hover_data=["Confirmed","Deaths","Recovered"],animation_frame="Date",color_continuous_scale="Portland",radius=7,zoom=0,height=700)
fig.update_layout(title='Worldwide Corona Virus Cases')
fig.update_layout(mapbox_style="open-street-map",mapbox_center_lon=0)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
df_recovered=pd.read_csv('/kaggle/input/dataset/time_series_covid_19_recovered.csv')
df_recovered.rename(columns={'Country/Region':'Country'},inplace=True)
df_main2=pd.merge(df,df_recovered,on=['Country','Province/State']) 
df_main2
fig=px.density_mapbox(df_main2,lat="Lat",lon="Long",hover_name="Province/State",hover_data=["Confirmed","Deaths","Recovered"],animation_frame="Date",color_continuous_scale="Portland",radius=7,zoom=0,height=700)
fig.update_layout(title='Worldwide Corona Virus Cases')
fig.update_layout(mapbox_style="open-street-map",mapbox_center_lon=0)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})