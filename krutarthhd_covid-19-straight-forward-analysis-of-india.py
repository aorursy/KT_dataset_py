import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_context("notebook")
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
py.init_notebook_mode(connected=True)
#from sklearn.model_selection import train_test_split
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
ageGroup = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
covid19India = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
hospitalBeds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
icmrTestLabs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
indiDetails = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
indiaCencus = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')
stateDetails = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')
plt.figure(figsize=(14,8))
sns.barplot(data=ageGroup,x='AgeGroup',y='TotalCases',color=sns.color_palette('Set3')[0])
plt.title('Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Total Cases')
for i in range(ageGroup.shape[0]):
    count = ageGroup.iloc[i]['TotalCases']
    plt.text(i,count+1,ageGroup.iloc[i]['Percentage'],ha='center')
    
from IPython.display import display, Markdown
display(Markdown("Most Number of cases have occured in the age group **20-50**"))
plt.figure(figsize=(14,8))
sns.countplot(data=indiDetails,x='gender',order=indiDetails['gender'].value_counts().index,color=sns.color_palette('Set3')[2])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Total Cases')
order2 = indiDetails['gender'].value_counts()

for i in range(order2.shape[0]):
    count = order2[i]
    strt='{:0.1f}%'.format(100*count / indiDetails.gender.dropna().count() )
    plt.text(i,count+2,strt,ha='center')
indiDetails.gender.fillna('Missing',inplace = True)
plt.figure(figsize=(14,8))
sns.countplot(data=indiDetails,x='gender',order=indiDetails['gender'].value_counts().index,color=sns.color_palette('Set3')[1])
plt.title('Gender Distribution (Considering Missing Values)')
plt.xlabel('Gender')
plt.ylabel('Total Cases')
order2 = indiDetails['gender'].value_counts()

for i in range(order2.shape[0]):
    count = order2[i]
    strt='{:0.1f}%'.format(100*count / indiDetails.shape[0])
    plt.text(i,count+2,strt,ha='center')
covid19India.info()
covid19India['Date'] = pd.to_datetime(covid19India['Date'],dayfirst=True)
df1=covid19India.groupby('Date').sum()
df1.reset_index(inplace=True)
plt.figure(figsize= (14,8))
plt.xticks(rotation = 90 ,fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)

ax1 = plt.plot_date(data=df1,y= 'Confirmed',x= 'Date',label = 'Confirmed',linestyle ='-',color = 'b')
ax2 = plt.plot_date(data=df1,y= 'Cured',x= 'Date',label = 'Cured',linestyle ='-',color = 'g')
ax3 = plt.plot_date(data=df1,y= 'Deaths',x= 'Date',label = 'Death',linestyle ='-',color = 'r')
plt.legend();
df2=df1.tail(25)
df2['Date'] = df2['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
plt.figure(figsize=(14,8))
sns.barplot(data=df2,x='Date',y='Confirmed',color=sns.color_palette('Set3')[3],label='Confirmed')
sns.barplot(data=df2,x='Date',y='Cured',color=sns.color_palette('Set3')[4],label='Cured')
sns.barplot(data=df2,x='Date',y='Deaths',color=sns.color_palette('Set3')[5],label='Deaths')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation = 90)
plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)
plt.legend(frameon=True,fontsize=12);
state_cases=covid19India.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
state_cases['Active'] = state_cases['Confirmed'] - abs((state_cases['Deaths']- state_cases['Cured']))
state_cases["Death Rate (per 100)"] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
state_cases["Cure Rate (per 100)"] = np.round(100*state_cases["Cured"]/state_cases["Confirmed"],2)
state_cases.sort_values('Confirmed', ascending= False).fillna(0).style.background_gradient(cmap='Reds',subset=["Confirmed"])\
                        .background_gradient(cmap='Blues',subset=["Deaths"])\
                        .background_gradient(cmap='Greens',subset=["Cured"])\
                        .background_gradient(cmap='Purples',subset=["Active"])\
                        .background_gradient(cmap='Greys',subset=["Death Rate (per 100)"])\
                        .background_gradient(cmap='Oranges',subset=["Cure Rate (per 100)"])
fig = px.treemap(state_cases,path=['State/UnionTerritory'],values='Active',hover_data=['Confirmed','Deaths','Cured'],color='Active',
                 color_continuous_scale='Reds')
fig.show()
state_cases=state_cases.sort_values('Confirmed', ascending= False).fillna(0)
state_cases=state_cases.head(15)
state_cases
plt.figure(figsize=(14,8))
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Confirmed',color=sns.color_palette('Set3')[3],label='Confirmed')
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Active',color=sns.color_palette('Set3')[7],label='Active')
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Cured',color=sns.color_palette('Set3')[8],label='Cured')
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Deaths',color=sns.color_palette('Set3')[9],label='Deaths')
plt.xticks(rotation=90)
plt.legend();
df3=indiDetails.groupby(['detected_state','detected_district']).count()
df3.reset_index(inplace=True)
states_list=['Maharashtra','Gujarat','Delhi','Rajasthan','Madhya Pradesh','Tamil Nadu','Uttar Pradesh','Telangana','Andhra Pradesh',
            'West Bengal','Karnataka','Kerala','Jammu and Kashmir','Punjab','Haryana']
plt.figure(figsize=(20,60))
for i,state in enumerate(states_list):
    plt.subplot(8,2,i+1)
    df4=df3[df3['detected_state']==state].sort_values('id',ascending=False)
    df4=df4.head(10)
    sns.barplot(data=df4,x='id',y='detected_district')
    plt.xlabel('Number of Cases')
    plt.ylabel('')
    plt.title(state)
plt.tight_layout()
plt.show()
states_list=['Maharashtra','Gujarat','Delhi','Rajasthan','Madhya Pradesh','Tamil Nadu','Uttar Pradesh','Andhra Pradesh',
            'West Bengal','Karnataka','Kerala','Jammu and Kashmir','Punjab','Haryana']
df5=covid19India[covid19India['Date']>'2020-04-07']
df5=df5.groupby(['Date','State/UnionTerritory']).sum()
df5.reset_index(inplace=True)
df5['Date'] = df5['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
plt.figure(figsize=(20,60))

for i,state in enumerate(states_list):
    plt.subplot(7,2,i+1)
    df4=df5[df5['State/UnionTerritory']==state]
    plt.bar(df4.Date,df4.Confirmed,label='Confirmed')
    plt.bar(df4.Date,df4.Cured,label='Cured')
    plt.bar(df4.Date,df4.Deaths,label='Death')
    plt.xticks(rotation=90)
    plt.title(state)
    plt.ylabel('Total Cases')
    plt.xlabel('Date')
    plt.legend()
plt.tight_layout()
plt.show()
covid19India['Date'] = pd.to_datetime(covid19India['Date'],dayfirst=True)
data=covid19India.groupby(['Date','State/UnionTerritory'])['Confirmed','Cured','Deaths'].sum()
data.reset_index(inplace=True)
data['Date']=data['Date'].apply(lambda x: x.strftime('%d-%m-%Y'))

import json
import folium

state_cases=covid19India.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()

with open('../input/india-geojson-file/state/india_state.geojson') as file:
    geojsonData = json.load(file)
    
for i in geojsonData['features']:
    if(i['properties']['NAME_1']=='Orissa'):
        i['properties']['NAME_1']='Odisha'
    elif(i['properties']['NAME_1']=='Uttaranchal'):
        i['properties']['NAME_1']='Uttarakhand'
    
for i in geojsonData['features']:
    i['id'] = i['properties']['NAME_1']
    
data = state_cases

map_choropleth = folium.Map(location = [20.5937,78.9629], zoom_start = 4)

folium.Choropleth(geo_data=geojsonData,
                 data=data,
                 name='CHOROPLETH',
                 key_on='feature.id',
                 columns = ['State/UnionTerritory','Confirmed'],
                 fill_color='YlOrRd',
                 fill_opacity=0.7,
                 line_opacity=0.8,
                 bins=[0,100,500,1000,3000,5000,10000,20000,60000,150000],
                 legend_name='Confirmed Cases',
                 highlight=True).add_to(map_choropleth)

folium.LayerControl().add_to(map_choropleth)

display(map_choropleth)

#gujaratDetail=indiDetails[indiDetails['detected_state']=='Gujarat']
#gujaratDetail=gujaratDetail.groupby('detected_district').count()
#ujaratDetail.reset_index(inplace=True)

#with open('../input/gujarat-geojson-districts-file/gujarat.geojson') as file:
#    geojsonData = json.load(file)
    
#for i in geojsonData['features']:
#    i['id'] = i['properties']['NAME_2']
    
#data = gujaratDetail

#map_choropleth = folium.Map(location = [22.2587, 71.1924], zoom_start = 7)
#folium.Choropleth(geo_data=geojsonData,
#                data=data,
#                 name='CHOROPLETH',
#                 key_on='feature.id',
#                 columns = ['detected_district','diagnosed_date'],
#                 fill_color='YlOrRd',
#                fill_opacity=0.7,
#                 line_opacity=0.8,
#                 legend_name='Population',
#                 bins=[0,20,100,1000,5000],
#                 highlight=True).add_to(map_choropleth)
#
#folium.LayerControl().add_to(map_choropleth)
#
#map_choropleth
utm = pd.read_csv('../input/utm-of-india/UTM ZONES of INDIA.csv')
dataLoc=covid19India.merge(utm , left_on='State/UnionTerritory', right_on='State / Union Territory')
dataLoc.drop(columns=['State / Union Territory'],inplace=True)
data2=dataLoc.groupby(['Date','State/UnionTerritory','Latitude','Longitude'])['Confirmed','Cured','Deaths'].sum()
data2.reset_index(inplace=True)
data2 = data2[data2['Date']>'2020-04-02']
data2['Date']=data2['Date'].apply(lambda x: x.strftime('%d-%m-%Y'))
data2['size'] = data2['Confirmed']*100000000
fig = px.scatter_mapbox(data2, lat="Latitude", lon="Longitude",
                     color="Confirmed", size='size',hover_data=['State/UnionTerritory'],
                     color_continuous_scale='Oranges', animation_frame="Date", 
                     title='Spread total cases over time in India')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, mapbox_center = {"lat":20.5937,"lon":78.9629})
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()
import IPython
IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977187" data-url="https://flo.uri.sh/visualisation/1977187/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
fig = px.treemap(icmrTestLabs, path=['state','city'],
                  color='city', hover_data=['lab','address'],
                  color_continuous_scale='Purples')
fig.show()
state=list(icmrTestLabs['state'].value_counts().index)
count=list(icmrTestLabs['state'].value_counts())
plt.figure(figsize=(14,8))
sns.barplot(x=count,y=state,color=sns.color_palette('Set3')[10])
plt.xlabel('Counts')
plt.ylabel('States')
plt.title('ICMR Test labs per States')
plt.tight_layout()
hospitalBeds.drop(labels=36,inplace=True)
plt.figure(figsize=(20,60))
plt.subplot(4,1,1)
hospitalBeds=hospitalBeds.sort_values('NumUrbanHospitals_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanHospitals_NHP18',color=sns.color_palette('Pastel2')[0])
plt.title('Urban Hospitals per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumUrbanHospitals_NHP18']
    plt.text(count+10,i,count,ha='center',va='center')

plt.subplot(4,1,2)
hospitalBeds=hospitalBeds.sort_values('NumRuralHospitals_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralHospitals_NHP18',color=sns.color_palette('Pastel2')[1])
plt.title('Rural Hospitals per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumRuralHospitals_NHP18']
    plt.text(count+100,i,count,ha='center',va='center')

plt.subplot(4,1,3)
hospitalBeds=hospitalBeds.sort_values('NumUrbanBeds_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanBeds_NHP18',color=sns.color_palette('Pastel2')[6])
plt.title('Rural Beds per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumUrbanBeds_NHP18']
    plt.text(count+1500,i,count,ha='center',va='center')

plt.subplot(4,1,4)
hospitalBeds=hospitalBeds.sort_values('NumRuralBeds_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralBeds_NHP18',color=sns.color_palette('Pastel2')[7])
plt.title('Rural Beds per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumRuralBeds_NHP18']
    plt.text(count+1500,i,count,ha='center',va='center')

plt.show()
plt.tight_layout()
pred = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
pred = pred[pred["Country/Region"]=="India"]
pred = pred.fillna(0)
predgrp = pred.groupby("Date")[["Confirmed","Recovered","Deaths"]].sum().reset_index()

#Model
pred_cnfrm = predgrp.loc[:,["Date","Confirmed"]]
pr_data = pred_cnfrm
pr_data.columns = ['ds','y']
m=Prophet()
m.fit(pr_data)
future=m.make_future_dataframe(periods=15)
forecast=m.predict(future)
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
