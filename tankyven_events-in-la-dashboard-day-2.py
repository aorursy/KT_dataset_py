import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast, json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import folium
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
df = pd.read_csv("../input/what's-happening-la-calendar-dataset/whats-happening-la-calendar-dataset.csv")
# Date pre-processing - Only include events after 2010
df['Start Year'] = df['Event Date & Time Start'].astype(str).apply(lambda x: x.split('-')[0])
df['End Year'] = df['Event Date & Time Ends'].astype(str).apply(lambda x: x.split('-')[0])
df = df[df['Start Year'].astype(float) > 2010]
df = df[df['End Year'].astype(float) > 2010]
df['Event Date & Time Start'] = pd.to_datetime(df['Event Date & Time Start'])
df['Event Date & Time Ends'] = pd.to_datetime(df['Event Date & Time Ends'])

# Add new date columns
df['Month'] = df['Event Date & Time Start'].apply(lambda time: time.month)
df['Day of Week'] = df['Event Date & Time Start'].apply(lambda time: time.dayofweek)
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df['Day of Week'] = pd.Categorical(df['Day of Week'], categories=['Mon','Tue','Wed','Thu','Fri','Sat', 'Sun'], ordered=True)

# Replace NAs in location address
df['Location Address'].fillna('{}', inplace = True)
# Normalize json in location address
def only_dict(d):
    '''
    Convert json string representation of dictionary to a python dict
    '''
    return ast.literal_eval(d)

location_data = json_normalize(data = df['Location Address'].apply(only_dict))
location_data['human_address'].fillna('{}', inplace = True)
human_address_data = json_normalize(data = location_data['human_address'].apply(only_dict))
# Events started in each year - bar plot
byStartYear = df.groupby('Start Year').count()
byStartYear = byStartYear.reset_index()

data = [go.Bar(
            x = byStartYear['Start Year'],
            y = byStartYear['Event Date & Time Start']
    )]
layout = go.Layout(
    title='Number of Events Started each Year',
)

fig = dict(data = data, layout = layout)
iplot(fig)
# Clean human_address data
human_address_data['city'].replace('', np.NaN, inplace = True)
human_address_data = human_address_data.dropna()

# Find number of events in each county from 2014 - 2017
bycity = human_address_data.groupby('city').count()
bycity = bycity.reset_index().sort_values(by='address', ascending = False).rename(str.capitalize,axis = 'columns')
bycity = bycity.rename(index = str, columns = {'Address': 'Count'})
# Bar chart showing top 5 cities where events were held
data = [go.Bar(
            x = bycity.head()['City'],
            y = bycity.head()['Count']
    )]
layout = go.Layout(
    title='Top 5 Cities for Events in LA',
)

fig = dict(data = data, layout = layout)
iplot(fig)
# Combine & clean data
df_address = df.reset_index().drop(columns = 'index').join(location_data)
df_address = df_address.sort_values(by = 'Event Date & Time Start', ascending = False)
df_address = df_address.dropna(subset=['Event Name', 'latitude', 'longitude']).reset_index().drop(columns = 'index')
# Create a geographical map of the 20 most recent events
LA_COORDINATES = (34.0522, -118.2437)

# for speed purposes
MAX_RECORDS = 20
  
# create empty map zoomed in on Los Angeles
map = folium.Map(location=LA_COORDINATES, zoom_start=10)
# add a marker for every record in the filtered data, use a clustered view
# folium.Marker([float(df_address["latitude"][12]),float(df_address["longitude"][12])],
#                   popup = df_address["Event Name"][12] + " Start: " + str(df_address['Event Date & Time Start'][12])).add_to(map)
for i in range(0,MAX_RECORDS):
    folium.Marker([float(df_address["latitude"][i]),float(df_address["longitude"][i])],
                  popup = df_address["Event Name"][i] + " - " + str(df_address['Event Description'][i])).add_to(map)
display(map)
