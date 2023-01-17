import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline
import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster

from subprocess import check_output
print(check_output(['ls','../input']).decode('utf8'))
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
print('The shape of the training data is:',train.shape)
print('The shape of the test data is :',test.shape)
train.head(5)
train.columns[train.isnull().any()]
train=pd.read_csv('../input/train.csv',parse_dates=['Dates'])
train['Year']=train['Dates'].dt.year
train['Month']=train['Dates'].dt.month
train['Hour']=train['Dates'].dt.hour
train.head()
def street_addr(x):
    street=x.split(' ')
    return (' '.join(street[-2:]))
train['addr']=train['Address'].apply(lambda x:street_addr(x))
train['addr'].head()
commo_crime=train['Category'].value_counts().sort_values(ascending=False).reset_index().head(10)
commo_crime.columns=['Crime','Count']
data = [go.Bar(
            x=commo_crime.Crime,
            y=commo_crime.Count,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')

train['PdDistrict'].value_counts()
commo_dis=train['PdDistrict'].value_counts().sort_values(ascending=False).reset_index().head(10)
commo_dis.columns=['District','Count']
data = [go.Bar(
            y=commo_dis.District,
            x=commo_dis.Count,
             opacity=0.6,
             orientation = 'h'
    )]

py.iplot(data, filename='basic-bar')

train['Year'].value_counts()
year_count=train['Year'].value_counts().reset_index().sort_values(by='index')
year_count.columns=['Year','Count']
# Create a trace
trace = go.Scatter(
    x = year_count.Year,
    y = year_count.Count
)

data = [trace]

py.iplot(data, filename='basic-line')

train['addr'].value_counts().head(10)
year_count=train['addr'].value_counts().reset_index().sort_values(by='index').head(10)
year_count.columns=['addr','Count']
# Create a trace
tag = (np.array(year_count.addr))
sizes = (np.array((year_count['Count'] / year_count['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top Address with Most Crimes')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")

month=train['Month'].value_counts().sort_values(ascending=False).reset_index()
month.columns=['Month','Count']
data = [go.Bar(
            x=month.Month,
            y=month.Count,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')

data=[]
for i in range(2003,2015):
    year=train[train['Year']==i]
    year_count=year['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')
    
val=train['PdDistrict'].value_counts().reset_index()
val.columns=['District','Count']
x=val.District
data=[]
for i in x:
    district=train[train['PdDistrict']==i]
    year_count=district['Year'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Year','Count']
    trace = go.Scatter(
    x = year_count.Year,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')
m = folium.Map(
    location=[train.Y.mean(), train.X.mean()],
    tiles='Cartodb Positron',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Crime Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = train.Y.values[k], train.X.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = train.addr.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m
M= folium.Map(location=[train.Y.mean(), train.X.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 13) 

# List comprehension to make out list of lists
heat_data = [[[row['Y'],row['X']] 
                for index, row in train.head(1000).iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(M)

hm.save('world MPI heatmap.html')

# Display the map
M
top_crime=train['Category'].value_counts().reset_index().head(5)
top_crime
cat='LARCENY/THEFT'
new=train[train['Category']==cat]
m = folium.Map(
    location=[train.Y.mean(), train.X.mean()],
    tiles='Cartodb Positron',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Crime Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = new.Y.values[k], new.X.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = new.addr.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m
cat='OTHER OFFENSES'
new=train[train['Category']==cat]
m = folium.Map(
    location=[train.Y.mean(), train.X.mean()],
    tiles='Cartodb Positron',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Crime Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = new.Y.values[k], new.X.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = new.addr.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m
new=train[train['Category']=='NON-CRIMINAL']
M= folium.Map(location=[train.Y.mean(), train.X.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 13) 

# List comprehension to make out list of lists
heat_data = [[[row['Y'],row['X']] 
                for index, row in new.head(1000).iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(M)

hm.save('SanFran heatmap.html')

# Display the map
M
new=train[train['Category']=='ASSAULT']
M= folium.Map(location=[train.Y.mean(), train.X.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 13) 

# List comprehension to make out list of lists
heat_data = [[[row['Y'],row['X']] 
                for index, row in new.head(1000).iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(M)

hm.save('SanFran heatmap.html')

# Display the map
M