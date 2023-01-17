# upload your kaggle API token (you can get that from your account)
from google.colab import files
uploaded = files.upload()
# Importing libraries
%matplotlib inline

# Run this to create a kaggle environment
# !pip install -q kaggle
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 /root/.kaggle/kaggle.json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get data from kaggle 
import zipfile
# Download data
!kaggle datasets download -d imdevskp/corona-virus-report/
!kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset/
!kaggle datasets download -d imdevskp/covid19-corona-virus-india-dataset
!kaggle datasets download -d sudalairajkumar/covid19-in-italy/
# UnZip data
zip_ref = zipfile.ZipFile("corona-virus-report.zip", 'r')
zip_ref.extractall()
zip_ref = zipfile.ZipFile("novel-corona-virus-2019-dataset.zip", 'r')
zip_ref.extractall()
zip_ref = zipfile.ZipFile("covid19-corona-virus-india-dataset.zip", 'r')
zip_ref.extractall()
zip_ref = zipfile.ZipFile("covid19-in-italy.zip", 'r')
zip_ref.extractall()
zip_ref.close()
!ls
# load the data
data = pd.read_csv("covid_19_clean_complete.csv")
data.head(5)
# Replacing all the NaN values with Country/Region
data["Province/State"].fillna(data["Country/Region"], inplace=True)
# Get the latest data of every country/city
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.strftime('%m/%d/%Y')
data1 = data.sort_values('Date')
data = data1.loc[data1["Date"]==max(data1['Date'])][['Date',"Lat","Long",'Province/State',"Country/Region","Confirmed","Deaths","Recovered"]]

# lets also add active cases in every country
data["Active"] = data["Confirmed"] - data["Deaths"] - data["Recovered"]
data.head(5)

latest_data_aggregated = data.groupby(["Date","Province/State","Country/Region"],as_index=False)["Date","Lat","Long","Province/State","Country/Region","Confirmed","Deaths","Recovered","Active"].sum()
us_data = pd.read_csv("covid_19_data.csv")
# Get the latest data of every country/city
us_data['Last Update'] = pd.to_datetime(us_data['Last Update'])
us_data['Last Update'] = us_data['Last Update'].dt.strftime('%m/%d/%Y')
us_data1 = us_data.sort_values('Last Update')
us_data = us_data1.loc[us_data1["Last Update"]==max(us_data1['Last Update'])][['Last Update','Province/State',"Country/Region","Confirmed","Deaths","Recovered"]]

# lets also add active cases in every country
us_data["Active"] = us_data["Confirmed"] - us_data["Deaths"] - us_data["Recovered"]
us_data = us_data[us_data["Country/Region"]=="US"]
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="us lat and long")
lat, long_data = [],[]
d={}
for i in list(us_data['Province/State']):
  location = geolocator.geocode(i)
  try:
    d[i] = {"lat":location.latitude,"long":location.longitude}
  except:
    d[i] = {"lat":0,"long":0}
us_data["Lat"] = 0
us_data["Long"] = 0
us_data1["Lat"] = 0
us_data1["Long"] = 0

for i in list(us_data['Province/State']):
  us_data.loc[us_data['Province/State'] == i, 'Lat'] = d[i]["lat"]
  us_data.loc[us_data['Province/State'] == i, 'Long'] = d[i]["long"]
  us_data1.loc[us_data1['Province/State'] == i, 'Lat'] = d[i]["lat"]
  us_data1.loc[us_data1['Province/State'] == i, 'Long'] = d[i]["long"]
import folium
latitude = 39.91666667
longitude = 116.383333
world_map = folium.Map(location=[latitude, longitude], zoom_start=3.5,tiles='cartodbpositron')
for lat, lon, Confirmed,Deaths,Recovered,Active,name in zip(us_data['Lat'], us_data['Long'], us_data['Confirmed'],us_data['Deaths'],us_data['Recovered'],us_data['Active'], us_data['Province/State']):
    folium.CircleMarker([lat, lon],
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(Confirmed) + '<br>'
                                '<strong>Active Cases</strong>: ' + str(Active) +'<br>'
                                '<strong>Recovered Cases</strong>: ' + str(Recovered) +'<br>'
                                '<strong>Deaths Cases</strong>: ' + str(Deaths) +'<br>'),
                        color='red',
                        
                        fill_color='#e84545',
                        radius=10).add_to(world_map)
world_map
china_data = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="China"]
import folium
latitude = 39.91666667
longitude = 116.383333
world_map = folium.Map(location=[latitude, longitude], zoom_start=3.5,tiles='cartodbpositron')
for lat, lon, Confirmed,Deaths,Recovered,Active,name in zip(china_data['Lat'], china_data['Long'], china_data['Confirmed'],china_data['Deaths'],china_data['Recovered'],china_data['Active'], china_data['Province/State']):
    folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(Confirmed) + '<br>'
                                '<strong>Active Cases</strong>: ' + str(Active) +'<br>'
                                '<strong>Recovered Cases</strong>: ' + str(Recovered) +'<br>'
                                '<strong>Deaths Cases</strong>: ' + str(Deaths) +'<br>'),
                        color='red',
                        
                        fill_color='#e84545',
                        fill_opacity=0.7 ).add_to(world_map)
world_map
# Get the latest data of every country/city
italy_Data = pd.read_csv("covid19_italy_region.csv")
italy_Data['Date'] = pd.to_datetime(italy_Data['Date'])
italy_Data['Date'] = italy_Data['Date'].dt.strftime('%m/%d/%Y')
italy_Data1 = italy_Data.sort_values('Date')
italy_Data = italy_Data1.loc[italy_Data1["Date"]==max(italy_Data1['Date'])][['Date',"Latitude","Longitude",'RegionName',"TotalPositiveCases","Deaths","Recovered"]]

# lets also add active cases in every country
italy_Data["Active"] = italy_Data["TotalPositiveCases"] - italy_Data["Deaths"] - italy_Data["Recovered"]
italy_Data.rename(columns={'Latitude':'Lat'}, inplace=True)
italy_Data.rename(columns={'Longitude':'Long'}, inplace=True)
italy_Data.rename(columns={'TotalPositiveCases':'Confirmed'}, inplace=True)
import folium
latitude = 39.91666667
longitude = 116.383333
world_map = folium.Map(location=[latitude, longitude], zoom_start=3.5,tiles='cartodbpositron')
for lat, lon, Confirmed,Deaths,Recovered,Active,name in zip(italy_Data['Lat'], italy_Data['Long'], italy_Data['Confirmed'],italy_Data['Deaths'],italy_Data['Recovered'],italy_Data['Active'], italy_Data['RegionName']):
    folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(Confirmed) + '<br>'
                                '<strong>Active Cases</strong>: ' + str(Active) +'<br>'
                                '<strong>Recovered Cases</strong>: ' + str(Recovered) +'<br>'
                                '<strong>Deaths Cases</strong>: ' + str(Deaths) +'<br>'),
                        color='red',
                        
                        fill_color='#e84545' ).add_to(world_map)
world_map
india_data = pd.read_csv("complete.csv")
# Get the latest data of every country/city
india_data['Date'] = pd.to_datetime(india_data['Date'])
india_data['Date'] = india_data['Date'].dt.strftime('%m/%d/%Y')
india_data1 = india_data.sort_values('Date')
india_data = india_data1.loc[india_data1["Date"]==max(india_data1['Date'])][['Date','Name of State / UT',"Total Confirmed cases (Indian National)","Total Confirmed cases ( Foreign National )","Cured/Discharged/Migrated","Latitude","Longitude","Death"]]

# lets also add active cases in every country
india_data["Confirmed"] = india_data["Total Confirmed cases (Indian National)"] + india_data["Total Confirmed cases ( Foreign National )"]
india_data["Active"] = india_data["Confirmed"] - india_data["Death"] - india_data['Cured/Discharged/Migrated']
india_data1["Confirmed"] = india_data1["Total Confirmed cases (Indian National)"] + india_data1["Total Confirmed cases ( Foreign National )"]
india_data1["Active"] = india_data1["Confirmed"] - india_data1["Death"] - india_data1['Cured/Discharged/Migrated']

india_data.rename(columns={'Latitude':'Lat','Longitude':"Long",'Cured/Discharged/Migrated':'Cured'}, inplace=True)
import folium
latitude = 39.91666667
longitude = 116.383333
world_map = folium.Map(location=[latitude, longitude], min_zoom=4, max_zoom=6, zoom_start=4 ,tiles='cartodbpositron')
for lat, lon, Confirmed,Deaths,Recovered,Active,name in zip(india_data['Lat'], india_data['Long'], india_data["Confirmed"],india_data['Death'],india_data['Cured'],india_data['Active'], india_data['Name of State / UT']):
    folium.CircleMarker([lat, lon],
                        fill='#e84545',
                        tooltip = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(Confirmed) + '<br>'
                                '<strong>Active Cases</strong>: ' + str(Active) +'<br>'
                                '<strong>Recovered Cases</strong>: ' + str(Recovered) +'<br>'
                                '<strong>Deaths Cases</strong>: ' + str(Deaths) +'<br>'),
                        color='red',
                        radius = 10).add_to(world_map)
world_map
fig = px.scatter_geo(us_data2, lat="Lat", lon="Long", color='Confirmed', size='Confirmed', projection="natural earth",
                     hover_name="Province/State", scope='world', animation_frame="Last Update", 
                     range_color=[0, max(us_data2['Confirmed'])])
fig.show()
data1["Active"] = data1["Confirmed"] - data1["Deaths"] - data1["Recovered"]
latest_data_aggregated = data1.groupby(["Date","Province/State","Country/Region"],as_index=False)["Date","Lat","Long","Province/State","Country/Region","Confirmed","Deaths","Recovered","Active"].sum()
china_data = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="China"]
fig = px.scatter_geo(china_data, lat="Lat", lon="Long", color='Province/State', size='Confirmed', projection="natural earth",
                     hover_name="Province/State", scope='asia', animation_frame="Date", center={'lat':20, 'lon':78}, 
                     range_color=[0, max(china_data['Confirmed'])])
fig.show()
# lets also add active cases in every country
italy_Data1["Active"] = italy_Data1["TotalPositiveCases"] - italy_Data1["Deaths"] - italy_Data1["Recovered"]
fig = px.scatter_geo(italy_Data1, lat="Latitude", lon="Longitude", color='RegionName', size='TotalPositiveCases', projection="natural earth",
                     hover_name="RegionName", scope='europe', animation_frame="Date", 
                     range_color=[0, max(italy_Data1['TotalPositiveCases'])])
fig.show()
fig = px.scatter_geo(india_data1, lat="Latitude", lon="Longitude", color='Confirmed', size='Confirmed', projection="natural earth",
                     hover_name="State/UT", scope='asia', animation_frame="Date", center={'lat':20, 'lon':78}, 
                     range_color=[0, max(india_data1['Confirmed'])])
fig.show()
latest_data = data1.loc[data1["Date"]==max(data1['Date'])][['Date',"Lat","Long",'Province/State',"Country/Region","Confirmed","Deaths","Recovered"]]
latest_data_aggregated = latest_data.groupby(["Date","Province/State","Country/Region"],as_index=False)["Date","Lat","Long","Province/State","Country/Region","Confirmed","Deaths","Recovered"].sum()
china_data = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="China"]
from plotly.subplots import make_subplots

# Confirmed cases in China
fig_c = px.bar(china_data.sort_values('Confirmed', ascending=False)[:15][::-1], 
             x='Confirmed', y='Province/State',text='Confirmed', orientation='h')
#fig.show()

# Confirmed cases in US

fig_d = px.bar(us_data.sort_values('Confirmed', ascending=False)[:15][::-1], 
             x='Confirmed', y='Province/State', text='Confirmed', orientation='h')
#fig.show()

# Confirmed cases in Italy
fig_r = px.bar(italy_Data.sort_values('Confirmed', ascending=False)[:15][::-1], 
             x='Confirmed', y='RegionName', text='Confirmed', orientation='h')
#fig.show()

# Confirmed cases in India
fig_a = px.bar(india_data.sort_values('Confirmed', ascending=False)[:15][::-1], 
             x='Confirmed', y='Name of State / UT', text='Confirmed', orientation='h')
#fig.show()


fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.08, horizontal_spacing=0.1,
                    subplot_titles=("Confirmed China", "Confirmed us", "Confirmed Italy", "Confirmed India"))
fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)
fig.add_trace(fig_r['data'][0], row=2, col=1)
fig.add_trace(fig_a['data'][0], row=2, col=2)
fig.update_layout(height=800, title_text="Top 15")
from plotly.subplots import make_subplots

# Deaths cases in China
fig_c = px.bar(china_data.sort_values('Deaths', ascending=False)[:15][::-1], 
             x='Deaths', y='Province/State',text='Deaths', orientation='h')
#fig.show()

# Deaths cases in US

fig_d = px.bar(us_data.sort_values('Deaths', ascending=False)[:15][::-1], 
             x='Deaths', y='Province/State', text='Deaths', orientation='h')
#fig.show()

# Deaths cases in Italy
fig_r = px.bar(italy_Data.sort_values('Deaths', ascending=False)[:15][::-1], 
             x='Deaths', y='RegionName', text='Deaths', orientation='h')
#fig.show()

# Deaths cases in India
fig_a = px.bar(india_data.sort_values('Death', ascending=False)[:15][::-1], 
             x='Death', y='Name of State / UT', text='Death', orientation='h')
#fig.show()


fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.08, horizontal_spacing=0.1,
                    subplot_titles=("Deaths China", "Deaths us", "Deaths Italy", "Deaths India"))
fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)
fig.add_trace(fig_r['data'][0], row=2, col=1)
fig.add_trace(fig_a['data'][0], row=2, col=2)
fig.update_layout(height=800, title_text="Top 15")
from plotly.subplots import make_subplots

# Recovered cases in China
fig_c = px.bar(china_data.sort_values('Recovered', ascending=False)[:15][::-1], 
             x='Recovered', y='Province/State',text='Recovered', orientation='h')
#fig.show()

# Recovered cases in US

fig_d = px.bar(us_data.sort_values('Recovered', ascending=False)[:15][::-1], 
             x='Recovered', y='Province/State', text='Recovered', orientation='h')
#fig.show()

# Recovered cases in Italy
fig_r = px.bar(italy_Data.sort_values('Recovered', ascending=False)[:15][::-1], 
             x='Recovered', y='RegionName', text='Recovered', orientation='h')
#fig.show()

# Deaths cases in India
fig_a = px.bar(india_data.sort_values('Cured', ascending=False)[:15][::-1], 
             x='Cured', y='Name of State / UT', text='Cured', orientation='h')
#fig.show()


fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.08, horizontal_spacing=0.1,
                    subplot_titles=("Recovered China", "Recovered us", "Recovered Italy", "Recovered India"))
fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)
fig.add_trace(fig_r['data'][0], row=2, col=1)
fig.add_trace(fig_a['data'][0], row=2, col=2)
fig.update_layout(height=800, title_text="Top 15")
latest_data_aggregated = data1.groupby(["Date","Country/Region"],as_index=False)["Date","Lat","Long","Province/State","Country/Region","Confirmed","Deaths","Recovered","Active"].sum()
china_data = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="China"]
us_data = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="US"]
italy_Data = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="Italy"]
india_data = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="India"]

# confirmed cases china
fig_c = px.line(china_data, x="Date", y="Confirmed")

# confirmed cases us
fig_d = px.line(us_data, x="Date", y="Confirmed")

# confirmed cases italy
fig_r = px.line(italy_Data, x="Date", y="Confirmed")

# confirmed cases india
fig_a = px.line(india_data, x="Date", y="Confirmed")

fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.2, horizontal_spacing=0.1,
                    subplot_titles=("confirmed cases china", "confirmed cases us", "confirmed cases italy", "confirmed cases india"))
fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)
fig.add_trace(fig_r['data'][0], row=2, col=1)
fig.add_trace(fig_a['data'][0], row=2, col=2)
fig.update_layout(height=800, title_text="Worwide data over time")
# confirmed cases china
fig_c = px.line(china_data, x="Date", y="Confirmed", log_y=True, title="confirmed cases china")
fig_c.show()
# confirmed cases us
fig_d = px.line(us_data, x="Date", y="Confirmed", log_y=True, title="confirmed cases us")
fig_d.show()
# confirmed cases italy
fig_r = px.line(italy_Data, x="Date", y="Confirmed", log_y=True, title="confirmed cases italy")
fig_r.show()
# confirmed cases india
fig_a = px.line(india_data, x="Date", y="Confirmed", log_y=True, title="confirmed cases india")
fig_a.show()
# confirmed cases china
fig_c = px.bar(china_data, x="Date", y="Confirmed", barmode='group',
             height=400)

# Dconfirmed cases US
fig_d = px.bar(us_data, x="Date", y="Deaths", barmode='group',
             height=400)

# confirmed cases Italy
fig_r = px.bar(italy_Data, x="Date", y="Recovered", barmode='group',
             height=400)

# confirmed cases India
fig_a = px.bar(india_data, x="Date", y="Active", barmode='group',
             height=400)

fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.2, horizontal_spacing=0.1,
                    subplot_titles=("Confirmed cases china", "Confirmed cases US", "Confirmed cases Italy", "Confirmed cases India"))
fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)
fig.add_trace(fig_r['data'][0], row=2, col=1)
fig.add_trace(fig_a['data'][0], row=2, col=2)
fig.update_layout(height=800, title_text="country data over time",plot_bgcolor='rgb(250, 242, 242)')
china_data
data = data1[data1['Date'] == max(data1['Date'])]
latest_data_aggregated = data.groupby(["Date","Province/State","Country/Region"],as_index=False)["Date","Lat","Long","Province/State","Country/Region","Confirmed","Deaths","Recovered","Active"].sum()
flg = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="China"]

flg['mortalityRate'] = round((flg['Deaths']/flg['Confirmed'])*100, 2)
temp = flg[flg['Confirmed']>100]
temp = temp.sort_values('mortalityRate', ascending=False)

fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],
             x = 'mortalityRate', y = 'Province/State', 
             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',
             color_discrete_sequence=['darkred']
            )
fig.show()
data = us_data1[us_data1['Last Update'] == max(us_data1['Last Update'])]
latest_data_aggregated = data.groupby(["Last Update","Province/State","Country/Region"],as_index=False)["Last Update","Lat","Long","Province/State","Country/Region","Confirmed","Deaths","Recovered"].sum()
flg = latest_data_aggregated[latest_data_aggregated["Country/Region"]=="US"]

flg['mortalityRate'] = round((flg['Deaths']/flg['Confirmed'])*100, 2)
temp = flg[flg['Confirmed']>100]
temp = temp.sort_values('mortalityRate', ascending=False)

fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],
             x = 'mortalityRate', y = 'Province/State', 
             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',
             color_discrete_sequence=['darkred']
            )
fig.show()
data = italy_Data1[italy_Data1['Date'] == max(italy_Data1['Date'])]
flg = data.groupby(["Date","RegionName"],as_index=False)["Date","RegionName","TotalPositiveCases","Deaths","Recovered"].sum()

flg['mortalityRate'] = round((flg['Deaths']/flg['TotalPositiveCases'])*100, 2)
temp = flg[flg['TotalPositiveCases']>100]
temp = temp.sort_values('mortalityRate', ascending=False)

fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],
             x = 'mortalityRate', y = 'RegionName', 
             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',
             color_discrete_sequence=['darkred']
            )
fig.show()
data = india_data1[india_data1['Date'] == max(india_data1['Date'])]
flg = data.groupby(["Date","State/UT"],as_index=False)["Date","State/UT","Confirmed","Death","Cured/Discharged/Migrated"].sum()

flg['mortalityRate'] = round((flg['Death']/flg['Confirmed'])*100, 2)
temp = flg[flg['Confirmed']>100]
temp = temp.sort_values('mortalityRate', ascending=False)

fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],
             x = 'mortalityRate', y = 'State/UT', 
             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',
             color_discrete_sequence=['darkred']
            )
fig.show()
import plotly.graph_objs as go

china_data = data1[data1['Date'] == max(data1['Date'])]
china_data = china_data[china_data["Country/Region"]=="China"]
temp = china_data.sort_values('Confirmed', ascending=False)[:15][::-1]
fig = go.Figure(data=[
    go.Bar(name='Active', y=temp['Province/State'], x=temp['Active'], orientation='h'),
    go.Bar(name='Deaths', y=temp['Province/State'], x=temp['Deaths'], orientation='h'),
    go.Bar(name='Recovered', y=temp['Province/State'], x=temp['Recovered'], orientation='h')
])
# Change the bar mode
fig.update_layout(barmode='stack', height=900)
fig.update_traces(textposition='outside')
fig.show()
import plotly.graph_objs as go

us_data = us_data1[us_data1['Last Update'] == max(us_data1['Last Update'])]
us_data = us_data[us_data["Country/Region"]=="US"]
temp = us_data.sort_values('Confirmed', ascending=False)[:15][::-1]
fig = go.Figure(data=[
    go.Bar(name='Active', y=temp['Province/State'], x=temp['Confirmed'], orientation='h'),
    go.Bar(name='Deaths', y=temp['Province/State'], x=temp['Deaths'], orientation='h'),
    go.Bar(name='Recovered', y=temp['Province/State'], x=temp['Recovered'], orientation='h')
])
# Change the bar mode
fig.update_layout(barmode='stack', height=900)
fig.update_traces(textposition='outside')
fig.show()
import plotly.graph_objs as go

italy_Data = italy_Data1[italy_Data1['Date'] == max(italy_Data1['Date'])]
temp = italy_Data.sort_values('TotalPositiveCases', ascending=False)[:15][::-1]
fig = go.Figure(data=[
    go.Bar(name='Active', y=temp['RegionName'], x=temp['TotalPositiveCases'], orientation='h'),
    go.Bar(name='Deaths', y=temp['RegionName'], x=temp['Deaths'], orientation='h'),
    go.Bar(name='Recovered', y=temp['RegionName'], x=temp['Recovered'], orientation='h')
])
# Change the bar mode
fig.update_layout(barmode='stack', height=900)
fig.update_traces(textposition='outside')
fig.show()
import plotly.graph_objs as go

india_data = india_data1[india_data1['Date'] == max(india_data1['Date'])]
temp = india_data.sort_values('Confirmed', ascending=False)[:15][::-1]
fig = go.Figure(data=[
    go.Bar(name='Active', y=temp['State/UT'], x=temp['Confirmed'], orientation='h'),
    go.Bar(name='Deaths', y=temp['State/UT'], x=temp['Death'], orientation='h'),
    go.Bar(name='Recovered', y=temp['State/UT'], x=temp['Cured/Discharged/Migrated'], orientation='h')
])
# Change the bar mode
fig.update_layout(barmode='stack', height=900)
fig.update_traces(textposition='outside')
fig.show()
# Confirmed cases in china
fig = px.pie(china_data, values='Confirmed', names='Province/State', title='Confirmed cases worldwide')
fig.show()

# Recovered Cases in china
fig = px.pie(china_data, values='Recovered', names='Province/State', title='Recovered cases worldwide')
fig.show()

# Death cases in china
fig = px.pie(china_data, values='Deaths', names='Province/State', title='Death cases worldwide')
fig.show()
# Confirmed cases in US
fig = px.pie(us_data, values='Confirmed', names='Province/State', title='Confirmed cases worldwide')
fig.show()

# Recovered Cases in US
fig = px.pie(us_data, values='Recovered', names='Province/State', title='Recovered cases worldwide')
fig.show()

# Death cases in US
fig = px.pie(us_data, values='Deaths', names='Province/State', title='Death cases worldwide')
fig.show()
# Confirmed cases in Italy
fig = px.pie(italy_Data, values='Confirmed', names='RegionName', title='Confirmed cases worldwide')
fig.show()

# Recovered Cases in Italy
fig = px.pie(italy_Data, values='Recovered', names='RegionName', title='Recovered cases worldwide')
fig.show()

# Death cases in Italy
fig = px.pie(italy_Data, values='Deaths', names='RegionName', title='Death cases worldwide')
fig.show()
# Confirmed cases in India
fig = px.pie(india_data, values='Confirmed', names='State/UT', title='Confirmed cases worldwide')
fig.show()

# Recovered Cases in India
fig = px.pie(india_data, values='Cured/Discharged/Migrated', names='State/UT', title='Recovered cases worldwide')
fig.show()

# Death cases in India
fig = px.pie(india_data, values='Death', names='State/UT', title='Death cases worldwide')
fig.show()
data = data1[data1['Date'] == max(data1['Date'])]
china_data = data[data["Country/Region"]=="China"]
china_data = china_data.groupby("Date",as_index=False)["Confirmed","Deaths","Recovered","Active"].sum()
us_data = data[data["Country/Region"]=="US"]
italy_data = data[data["Country/Region"]=="Italy"]
india_data = data[data["Country/Region"]=="India"]
plt.pie([china_data["Active"], china_data["Deaths"], china_data["Recovered"]], labels=["Active","Deaths","Recovered"],
           autopct=None)
plt.show()
