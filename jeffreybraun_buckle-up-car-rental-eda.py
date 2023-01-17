import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
import os
init_notebook_mode()

df = pd.read_csv('/kaggle/input/cornell-car-rental-dataset/CarRentalData.csv')
df_fuelType = pd.DataFrame(df.fuelType.value_counts()).reset_index()
df_fuelType.rename(columns = {'index':'fuelType', 'fuelType':'count'}, inplace=True)

fig = px.pie(df_fuelType, values = 'count', names='fuelType', title = 'Fuel Type of Rented Cars')
fig.show()
print("Rating Statistics:")
print(df['rating'].describe())

fig = px.histogram(df, x = 'rating', title = 'Histogram of Rental Car Rating')
fig.show()

print("Renter Trips Taken Statistics:")
print(df['renterTripsTaken'].describe())

fig = px.histogram(df, x = 'renterTripsTaken', title = 'Histogram of Renter Trips Taken')
fig.show()
print("Review Count Statistics:")
print(df['reviewCount'].describe())

fig = px.histogram(df, x = 'reviewCount', title = 'Histogram of Review Count')
fig.show()
def get_average_lat_long(city, ltype):
    choices = df[df['location.city'] == city]
    lat = choices['location.latitude'].mean()
    long = choices['location.longitude'].mean()
    if ltype == 0:
        return lat
    else:
        return long

df_location = pd.DataFrame(df['location.city'].value_counts()).reset_index()
df_location.rename(columns = {'index':'city', 'location.city':'count'}, inplace=True)
df_location['latitude'] = df_location['city'].apply(lambda x: get_average_lat_long(x, 0))
df_location['longitude'] = df_location['city'].apply(lambda x: get_average_lat_long(x, 1))
df_location['text'] = df_location['city'] + '<br>Car Rentals ' + (df_location['count']).astype(str)

limits = [(0,2),(3,10),(11,50),(51,100),(101,1000)]
colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
cities = []
scale = 0.5

fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df_location[lim[0]:lim[1]]
    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['count']/scale,
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = 'Car Rentals by City<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

fig.show()

df_state = pd.DataFrame(df['location.state'].value_counts()).reset_index()
df_state.rename(columns = {'index':'state', 'location.state':'count'}, inplace=True)

fig = go.Figure(data=go.Choropleth(
    locations=df_state['state'], # Spatial coordinates
    z = df_state['count'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Number of Cars Rented",
))

fig.update_layout(
    title_text = 'Car Rentals by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
df_owner = pd.DataFrame(df['owner.id'].value_counts()).reset_index()
df_owner.rename(columns = {'index':'owner_id', 'owner.id':'number of rental cars'}, inplace=True)

print('Total Number of Unique Rental Cars per Owner Statistics:')
print(df_owner['number of rental cars'].describe())


fig = px.histogram(df_owner, x = 'number of rental cars', title='Total Number of Unique Rental Cars per Owner')
fig.show()
print('Daily Rate of Car Rental Statistics:')
print(df['rate.daily'].describe())

fig = px.histogram(df, x = 'rate.daily', title='Daily Rate of Car Rental')
fig.show()
df_make_model = df.groupby(['vehicle.make', 'vehicle.model']).size().reset_index()
df_make_model.rename(columns = {0:'count'}, inplace=True)
df_make_model.replace('Mercedes-benz', 'Mercedes-Benz', inplace=True)
df_make_model['make_count'] = df_make_model['vehicle.make'].apply(lambda x : df_make_model[df_make_model['vehicle.make'] == x]['count'].sum())
df_make_model.sort_values(by = 'make_count', ascending=False, inplace=True)

fig = px.bar(df_make_model[df_make_model['make_count'] >45], x = 'vehicle.make', y='count', color = 'vehicle.model', title='Make and Model of Top 25 Most Rented Cars')
fig.update_layout(showlegend = False)
fig.show()
df_vehicleType = pd.DataFrame(df['vehicle.type'].value_counts()).reset_index()
df_vehicleType.rename(columns = {'index':'vehicle.type', 'vehicle.type':'count'}, inplace=True)

fig = px.pie(df_vehicleType, values = 'count', names='vehicle.type', title = 'Vehicle Type of Rented Cars')
fig.show()
print('Vehicle Year Statistics:')
print(df['vehicle.year'].describe())

fig = px.histogram(df, x = 'vehicle.year', title='Year of Vehicle')
fig.show()
