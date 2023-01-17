import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


import plotly.express as px
import plotly.graph_objs as go

#Some styling
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

#Showing full path of datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Disable warnings 
import warnings
warnings.filterwarnings('ignore')

city_day = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")
city_day.head()
city_day.info()
#We convert Date object to datetime

city_day['Date'] = pd.to_datetime(city_day['Date'])
city_day.isna().sum()
#We will fill the NA values with 0 and AQI_Bucket columns missing values with Not Known
for col in city_day.columns:
    if city_day[col].dtype=="O":
        continue
    else:
        city_day[col]=city_day[col].fillna(city_day[col].mean())
city_day['AQI_Bucket'] = city_day['AQI_Bucket'].fillna("Not Known")
city_day.isna().sum()
print("The city_day data is available from {} to {}".format(city_day['Date'].min(),city_day['Date'].max()))
print("Cities covered under this dataset are {}\n".format(city_day['City'].nunique()))

print("Cities :- {}".format(city_day['City'].unique()))
city_day
city_day.describe()
city_day['AQI_Bucket'].value_counts()
city_day.columns
city_day['year'] = city_day['Date'].dt.year
city_day
'''
Which city has best air quality currently

barplot like green yellow displaying all pollutants
'''
city_day['year'] = city_day['year'].astype(str)
temp = city_day.groupby(['year'])[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 
                                                  'NH3', 'CO', 'SO2','O3', 'Benzene', 
                                                  'Toluene', 'Xylene', 'AQI']].median()
temp.style.background_gradient(cmap="Reds")
city_day['month'] = city_day['Date'].dt.month

import calendar

city_day['month'] = city_day['month'].apply(lambda x: calendar.month_abbr[x])

temp = city_day.groupby(['month'])[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']].median()
temp.style.background_gradient(cmap="Reds")
temp = city_day.groupby(['City'])[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']].median().reset_index()
for col in temp.columns[1:]:
    temp[col] = round(temp[col],2)
temp.head()
for col in temp.columns[1:]:
        fig = px.bar(temp.sort_values(by=col,ascending=False),x="City",y=col,color=col,text=col,title=col)
        fig.update_traces(textposition='outside')
        fig.show()
temp = city_day[city_day['Date']>="2020-01-20"]

temp = temp.groupby(['Date'])[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene',"AQI"]].median().reset_index()
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp["Date"],y=temp["PM2.5"],
                        mode='lines+markers',name="PM2.5"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["PM10"],
                        mode='lines+markers',name="PM10"))
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
    )
])
fig.update_layout(title={'text':'Change in PM2.5 and PM10 after 25th March'})
fig.update_xaxes(rangeslider_visible=True)
fig.show()

#Nitrogen Oxides
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp["Date"],y=temp["NO"],
                        mode='lines+markers',name="NO"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["NO2"],
                        mode='lines+markers',name="NO2"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["NOx"],
                        mode='lines+markers',name="NOx"))
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
    )
])
fig.update_layout(title={'text':'Change in nitrogne oxides levels after 25th March'})
fig.update_xaxes(rangeslider_visible=True)
fig.show()

#Ammonia,Carbon Monoxide and Sulphur Dioxide and Ozone
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp["Date"],y=temp["NH3"],
                        mode='lines+markers',name="NH3"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["CO"],
                        mode='lines+markers',name="CO"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["SO2"],
                        mode='lines+markers',name="SO2"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["O3"],
                        mode='lines+markers',name="O3"))
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
    )
])
fig.update_layout(title={'text':'Change in NH3,CO,SO2,O3 levels after 25th March'})
fig.update_xaxes(rangeslider_visible=True)
fig.show()

#Benzene,Toluene,Xylene
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp["Date"],y=temp["Benzene"],
                        mode='lines+markers',name="Benzene"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["Toluene"],
                        mode='lines+markers',name="Toluene"))
fig.add_trace(go.Scatter(x=temp["Date"],y=temp["Xylene"],
                        mode='lines+markers',name="Xylene"))
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
    )
])
fig.update_layout(title={'text':'Change in benzene,toluene,xylene levels after 25th March'})
fig.update_xaxes(rangeslider_visible=True)
fig.show()
#AQI
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp["Date"],y=temp["AQI"],
                        mode='lines+markers',name="AQI"))
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
    )
])
fig.update_layout(title={'text':'Change in Air Quality Index after 25th March'})
fig.update_xaxes(rangeslider_visible=True)
fig.show()
import copy
city_day_new = city_day.copy()
city_day_new['nitrogen_oxides'] = city_day_new["NO"]+city_day_new["NO2"]+city_day_new["NOx"]
city_day_new['BTX'] = city_day_new["Benzene"]+city_day_new["Toluene"]+city_day_new["Xylene"]
temp = city_day_new.query('City in ["Mumbai","Delhi","Bengaluru","Kolkata","Chennai","Hyderabad"]')
temp = temp.query('month in ["Jan","Feb","Mar","Apr"]')
temp = temp.query('year in ["2019","2020"]')
temp = temp.groupby(['City','year'])[['PM2.5', 'PM10', 'nitrogen_oxides', 'NH3', 'CO', 'SO2',
       'O3', 'BTX', 'AQI']].median().reset_index()
temp
mumbai = temp[temp['City']=="Mumbai"]
bengaluru = temp[temp['City']=="Bengaluru"]
chennai = temp[temp['City']=="Chennai"]
delhi = temp[temp['City']=="Delhi"]
kolkata = temp[temp['City']=="Kolkata"]
hyderabad = temp[temp['City']=="Hyderabad"]
def before_and_after_covid19(pollutants):
    for i in range(len(pollutants)):
        fig =make_subplots(rows=2,cols=3,subplot_titles=("Mumbai","Bengaluru","Chennai",
                                                    "Kolkata","Hyderabad","Delhi"))
        fig.add_trace(go.Bar(x=mumbai["year"],y=mumbai[pollutants[i]],
                            marker=dict(color=mumbai[pollutants[i]],coloraxis="coloraxis")),1,1)
        fig.add_trace(go.Bar(x=bengaluru["year"],y=bengaluru[pollutants[i]],
                            marker=dict(color=bengaluru[pollutants[i]],coloraxis="coloraxis")),1,2)
        fig.add_trace(go.Bar(x=chennai["year"],y=chennai[pollutants[i]],
                            marker=dict(color=chennai[pollutants[i]],coloraxis="coloraxis")),1,3)
        fig.add_trace(go.Bar(x=kolkata["year"],y=kolkata[pollutants[i]],
                            marker=dict(color=kolkata[pollutants[i]],coloraxis="coloraxis")),2,1)
        fig.add_trace(go.Bar(x=hyderabad["year"],y=hyderabad[pollutants[i]],
                            marker=dict(color=hyderabad[pollutants[i]],coloraxis="coloraxis")),2,2)
        fig.add_trace(go.Bar(x=delhi["year"],y=delhi[pollutants[i]],
                            marker=dict(color=delhi[pollutants[i]],coloraxis="coloraxis")),2,3)

        fig.update_layout(showlegend=False,coloraxis=dict(colorscale='ylgn'),height=800)
        fig.update_layout(title={'text':pollutants[i]})
        fig.show()
#We'll take the primary pollutants that are
pollutants = ['PM2.5', 'PM10', 'nitrogen_oxides','NH3','CO', 'SO2','O3','BTX']
before_and_after_covid19(pollutants)
fig =make_subplots(rows=2,cols=3,subplot_titles=("Mumbai","Bengaluru","Chennai",
                                            "Kolkata","Hyderabad","Delhi"))
fig.add_trace(go.Bar(x=mumbai["year"],y=mumbai["AQI"],
                    marker=dict(color=mumbai["AQI"],coloraxis="coloraxis")),1,1)
fig.add_trace(go.Bar(x=bengaluru["year"],y=bengaluru["AQI"],
                    marker=dict(color=bengaluru["AQI"],coloraxis="coloraxis")),1,2)
fig.add_trace(go.Bar(x=chennai["year"],y=chennai["AQI"],
                    marker=dict(color=chennai["AQI"],coloraxis="coloraxis")),1,3)
fig.add_trace(go.Bar(x=kolkata["year"],y=kolkata["AQI"],
                    marker=dict(color=kolkata["AQI"],coloraxis="coloraxis")),2,1)
fig.add_trace(go.Bar(x=hyderabad["year"],y=hyderabad["AQI"],
                    marker=dict(color=hyderabad["AQI"],coloraxis="coloraxis")),2,2)
fig.add_trace(go.Bar(x=delhi["year"],y=delhi["AQI"],
                    marker=dict(color=delhi["AQI"],coloraxis="coloraxis")),2,3)

fig.update_layout(showlegend=False,coloraxis=dict(colorscale='reds'),height=700)
fig.update_layout(title={'text':"AQI"})
fig.show()
temp = city_day_new[city_day_new['Date']>="2019-11-01"]
temp = temp.query('City in ["Mumbai","Delhi","Bengaluru","Kolkata","Chennai","Hyderabad"]')
temp = temp.query('month in ["Nov","Dec","Jan","Feb","Mar","Apr"]')
temp = temp.groupby(['City','month'])[['PM2.5', 'PM10', 'nitrogen_oxides', 'NH3', 'CO', 'SO2',
       'O3','BTX','AQI']].median().reset_index()
def before_and_after_lockdown(pollutants):
    mumbai = temp[temp['City']=="Mumbai"]
    bengaluru = temp[temp['City']=="Bengaluru"]
    chennai = temp[temp['City']=="Chennai"]
    delhi = temp[temp['City']=="Delhi"]
    kolkata = temp[temp['City']=="Kolkata"]
    hyderabad = temp[temp['City']=="Hyderabad"]
    for i in range(len(pollutants)):
        mumbai = mumbai.sort_values(by=pollutants[i],ascending=False)
        bengaluru = bengaluru.sort_values(by=pollutants[i],ascending=False)
        chennai = chennai.sort_values(by=pollutants[i],ascending=False)
        kolkata = kolkata.sort_values(by=pollutants[i],ascending=False)
        hyderabad = hyderabad.sort_values(by=pollutants[i],ascending=False)
        delhi = delhi.sort_values(by=pollutants[i],ascending=False)
        fig =make_subplots(rows=2,cols=3,subplot_titles=("Mumbai","Bengaluru","Chennai",
                                                    "Kolkata","Hyderabad","Delhi"))
        fig.add_trace(go.Bar(x=mumbai["month"],y=mumbai[pollutants[i]],
                            marker=dict(color=mumbai[pollutants[i]],coloraxis="coloraxis")),1,1)
        fig.add_trace(go.Bar(x=bengaluru["month"],y=bengaluru[pollutants[i]],
                            marker=dict(color=bengaluru[pollutants[i]],coloraxis="coloraxis")),1,2)
        fig.add_trace(go.Bar(x=chennai["month"],y=chennai[pollutants[i]],
                            marker=dict(color=chennai[pollutants[i]],coloraxis="coloraxis")),1,3)
        fig.add_trace(go.Bar(x=kolkata["month"],y=kolkata[pollutants[i]],
                            marker=dict(color=kolkata[pollutants[i]],coloraxis="coloraxis")),2,1)
        fig.add_trace(go.Bar(x=hyderabad["month"],y=hyderabad[pollutants[i]],
                            marker=dict(color=hyderabad[pollutants[i]],coloraxis="coloraxis")),2,2)
        fig.add_trace(go.Bar(x=delhi["month"],y=delhi[pollutants[i]],
                            marker=dict(color=delhi[pollutants[i]],coloraxis="coloraxis")),2,3)

        fig.update_layout(showlegend=False,coloraxis=dict(colorscale='ylgn'),height=800)
        fig.update_layout(title={'text':pollutants[i]})
        fig.show()
before_and_after_lockdown(pollutants)
mumbai = temp[temp['City']=="Mumbai"].sort_values(by="AQI",ascending=False)
bengaluru = temp[temp['City']=="Bengaluru"].sort_values(by="AQI",ascending=False)
chennai = temp[temp['City']=="Chennai"].sort_values(by="AQI",ascending=False)
delhi = temp[temp['City']=="Delhi"].sort_values(by="AQI",ascending=False)
kolkata = temp[temp['City']=="Kolkata"].sort_values(by="AQI",ascending=False)
hyderabad = temp[temp['City']=="Hyderabad"].sort_values(by="AQI",ascending=False)



fig =make_subplots(rows=2,cols=3,subplot_titles=("Mumbai","Bengaluru","Chennai",
                                            "Kolkata","Hyderabad","Delhi"))
fig.add_trace(go.Bar(x=mumbai["month"],y=mumbai["AQI"],
                    marker=dict(color=mumbai["AQI"],coloraxis="coloraxis")),1,1)
fig.add_trace(go.Bar(x=bengaluru["month"],y=bengaluru["AQI"],
                    marker=dict(color=bengaluru["AQI"],coloraxis="coloraxis")),1,2)
fig.add_trace(go.Bar(x=chennai["month"],y=chennai["AQI"],
                    marker=dict(color=chennai["AQI"],coloraxis="coloraxis")),1,3)
fig.add_trace(go.Bar(x=kolkata["month"],y=kolkata["AQI"],
                    marker=dict(color=kolkata["AQI"],coloraxis="coloraxis")),2,1)
fig.add_trace(go.Bar(x=hyderabad["month"],y=hyderabad["AQI"],
                    marker=dict(color=hyderabad["AQI"],coloraxis="coloraxis")),2,2)
fig.add_trace(go.Bar(x=delhi["month"],y=delhi["AQI"],
                    marker=dict(color=delhi["AQI"],coloraxis="coloraxis")),2,3)

fig.update_layout(showlegend=False,coloraxis=dict(colorscale='reds'),height=700)
fig.update_layout(title={'text':"AQI"})
fig.show()
city_day_2020 = city_day_new[city_day_new['year']=="2020"]
city_day_2020
mumbai = city_day_2020[city_day_2020['City']=="Mumbai"]
bengaluru = city_day_2020[city_day_2020['City']=="Bengaluru"]
hyderabad = city_day_2020[city_day_2020['City']=="Hyderabad"]
kolkata = city_day_2020[city_day_2020['City']=="Kolkata"]
chennai = city_day_2020[city_day_2020['City']=="Chennai"]
delhi = city_day_2020[city_day_2020['City']=="Delhi"]
def before_and_after_lockdown_datewise(pollutants):
    mumbai = city_day_2020[city_day_2020['City']=="Mumbai"]
    bengaluru = city_day_2020[city_day_2020['City']=="Bengaluru"]
    hyderabad = city_day_2020[city_day_2020['City']=="Hyderabad"]
    kolkata = city_day_2020[city_day_2020['City']=="Kolkata"]
    chennai = city_day_2020[city_day_2020['City']=="Chennai"]
    delhi = city_day_2020[city_day_2020['City']=="Delhi"]
    for i in range(len(pollutants)):
        fig =make_subplots(rows=6,cols=1,subplot_titles=("Mumbai","Bengaluru","Chennai",
                                                    "Kolkata","Hyderabad","Delhi"))
        fig.add_trace(go.Bar(x=mumbai["Date"],y=mumbai[pollutants[i]],
                            marker=dict(color=mumbai[pollutants[i]],coloraxis="coloraxis")),1,1)
        fig.add_trace(go.Bar(x=bengaluru["Date"],y=bengaluru[pollutants[i]],
                            marker=dict(color=bengaluru[pollutants[i]],coloraxis="coloraxis")),2,1)
        fig.add_trace(go.Bar(x=chennai["Date"],y=chennai[pollutants[i]],
                            marker=dict(color=chennai[pollutants[i]],coloraxis="coloraxis")),3,1)
        fig.add_trace(go.Bar(x=kolkata["Date"],y=kolkata[pollutants[i]],
                            marker=dict(color=kolkata[pollutants[i]],coloraxis="coloraxis")),4,1)
        fig.add_trace(go.Bar(x=hyderabad["Date"],y=hyderabad[pollutants[i]],
                            marker=dict(color=hyderabad[pollutants[i]],coloraxis="coloraxis")),5,1)
        fig.add_trace(go.Bar(x=delhi["Date"],y=delhi[pollutants[i]],
                            marker=dict(color=delhi[pollutants[i]],coloraxis="coloraxis")),6,1)

        fig.update_layout(showlegend=False,coloraxis=dict(colorscale='ylgn'))
        fig.update_layout(title={'text':pollutants[i]+" Levels"})
        fig.update_layout(shapes=[
            dict(
              type= 'line',
              yref= 'paper', y0= 0, y1= 1,
              xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
            )
            ])
        fig.show()
before_and_after_lockdown_datewise(pollutants)
mumbai = city_day_2020[city_day_2020['City']=="Mumbai"]
bengaluru = city_day_2020[city_day_2020['City']=="Bengaluru"]
hyderabad = city_day_2020[city_day_2020['City']=="Hyderabad"]
kolkata = city_day_2020[city_day_2020['City']=="Kolkata"]
chennai = city_day_2020[city_day_2020['City']=="Chennai"]
delhi = city_day_2020[city_day_2020['City']=="Delhi"]

fig =make_subplots(rows=6,cols=1,subplot_titles=("Mumbai","Bengaluru","Chennai",
                                            "Kolkata","Hyderabad","Delhi"))
fig.add_trace(go.Bar(x=mumbai["Date"],y=mumbai["AQI"],
                    marker=dict(color=mumbai["AQI"],coloraxis="coloraxis")),1,1)
fig.add_trace(go.Bar(x=bengaluru["Date"],y=bengaluru["AQI"],
                    marker=dict(color=bengaluru["AQI"],coloraxis="coloraxis")),2,1)
fig.add_trace(go.Bar(x=chennai["Date"],y=chennai["AQI"],
                    marker=dict(color=chennai["AQI"],coloraxis="coloraxis")),3,1)
fig.add_trace(go.Bar(x=kolkata["Date"],y=kolkata["AQI"],
                    marker=dict(color=kolkata["AQI"],coloraxis="coloraxis")),4,1)
fig.add_trace(go.Bar(x=hyderabad["Date"],y=hyderabad["AQI"],
                    marker=dict(color=hyderabad["AQI"],coloraxis="coloraxis")),5,1)
fig.add_trace(go.Bar(x=delhi["Date"],y=delhi["AQI"],
                    marker=dict(color=delhi["AQI"],coloraxis="coloraxis")),6,1)

fig.update_layout(showlegend=False,coloraxis=dict(colorscale='ylgn'))
fig.update_layout(title={'text':"AQI"+" Levels"})
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
    )
    ])
fig.show()
data = [['Ahmedabad',23.0225,72.5714],
    ['Amravati',20.9320,77.7523],
    ['Bengaluru',12.9716,77.5946],
    ['Bhopal',23.2599,77.4126],
    ['Brajrajnagar',21.8285,83.9176],
    ['Gurugram',28.4595,77.0266],
    ['Jorapokhar',23.7041,86.4137],
    ['Mumbai',19.0760,72.8777],
    ['Shillong',25.5788,91.8933],
    ['Talcher',20.9501,85.2168],
    ['Hyderabad',17.389716,78.466689],
    ['Patna',25.402971,85.319843],
    ['Chandigarh',30.742600,76.758725],
    ['Delhi',28.645944,77.128045],
    ['Thiruvananthapuram',8.537102,76.887115],
    ['Bhopal',23.480726,77.392466],
    ['Aizawl',23.907813,92.853595],
    ['Amritsar',31.548680,74.798839],
    ['Jaipur',27.033459,75.771173],
    ['Chennai',13.058099,80.281392],
    ['Lucknow',26.838234,80.897060],
    ['Kolkata',22.550581,88.352765]]

coor= pd.DataFrame(data,columns=["City","Lat","Long"])
coor
city_day['City'].unique()
city_map = pd.merge(city_day,coor,on=['City'],how='inner')
city_map
#Before Lockdown
before = city_map[city_map['year']=="2020"]
before = before[before['Date']<="2020-03-25"]
before = before[['City',"Lat","Long","AQI"]]
before = before.groupby(['City',"Lat","Long"],as_index=False)[['AQI']].median()

#After lcokdown
after = city_map[city_map['Date']>"2020-03-25"]
after = after[['City',"Lat","Long","AQI"]]
after = after.groupby(['City',"Lat","Long"],as_index=False)[['AQI']].median()

fig = px.scatter_mapbox(before,
                        lat="Lat",
                        lon="Long",
                        color='AQI',
                        mapbox_style='open-street-map',
                        hover_name='City',
                        size='AQI',
                        center={"lat": 20.5937, "lon": 78.9629},
                        zoom=3.5,
                        hover_data=['City','AQI'],
                        title= 'AQI Levels before lockdown')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(template='plotly_dark',margin=dict(l=20,r=20,t=40,b=20))
fig.show()


fig = px.scatter_mapbox(after,
                        lat="Lat",
                        lon="Long",
                        color='AQI',
                        mapbox_style='open-street-map',
                        hover_name='City',
                        size='AQI',
                        center={"lat": 20.5937, "lon": 78.9629},
                        zoom=3.5,
                        hover_data=['City','AQI'],
                        title= 'AQI Levels after lockdown')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(template='plotly_dark',margin=dict(l=20,r=20,t=40,b=20))
fig.show()