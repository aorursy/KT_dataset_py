import numpy as np
import pandas as pd
import plotly.express as px
import folium
%matplotlib inline
df = pd.read_csv('/kaggle/input/ontarioproperties/properties.csv')
df.shape
# Removing houses with invalid lat and lng
df = df[(df['lat'] != -999 ) & (df['lng'] != -999)]
df.shape
fig = px.violin(df, y='Price ($)')
fig.show()
df = df[(df['Price ($)'] < 1000000 )]

fig = px.violin(df, y='Price ($)')
fig.show()
quantile = df.quantile([0,0.25,0.5,0.75,1])
quantile['Price ($)']
# Threshold
df = df[(df['Price ($)'] <= df.quantile(0.75)['Price ($)']) & (df['Price ($)'] >= df.quantile(0.25)['Price ($)'])]
df = df.sort_values('Price ($)', ascending = 0)
df.shape
fig = px.violin(df, y='Price ($)')
fig.show()
df.head()
df.tail()
# Mean Price by AreaName
dfGrouped = df.groupby(['AreaName']).filter(lambda x: len(x) > 5)
meanPrices = dfGrouped.groupby(['AreaName']).mean()
meanPrices['Size'] = df.groupby(['AreaName']).size()
meanPrices = meanPrices.sort_values('Price ($)', ascending = 0)
meanPrices.shape
meanPrices.head()
meanPrices.tail()
#mapPlot = folium.Map(location= [(df.lat.min() + df.lat.max()) / 2.0 , (df.lng.min() + df.lng.max()) / 2.0], zoom_start = 8)
mapPlot = folium.Map(location= [df.lat.mean(), df.lng.mean()], zoom_start = 8)
minPrice = meanPrices['Price ($)'].min()
maxPrice = meanPrices['Price ($)'].max()
range = maxPrice - minPrice
colormap = ['lightgray', 'gray', 'blue', 'green', 'orange', 'pink', 'lightred', 'red', 'black']
for index, row in meanPrices.iterrows(): 
    # Set icon color by price
    color = colormap[int(round((len(colormap) - 1) *  float(row['Price ($)'] - minPrice) / range, 0)) ]
    # Create a marker text with area name and mean price
    markerText =  str(index) + ' (' + str(int(row['Size'])) + ')' + ' ${:,.0f}'.format(row['Price ($)'])    
    folium.Marker([row['lat'], row['lng']], popup = markerText, icon=folium.Icon(color = color)).add_to(mapPlot)

mapPlot