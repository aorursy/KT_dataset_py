import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go
import plotly as py
import plotly.offline as py

#To view visualizations offline
py.initnotebookmode(connected=True)

# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Import the plotly graphobjects library 
import plotly.graph_objects as go

fig = go.Figure(go.Scattergeo())
fig.update_layout(height=300, margin={"r":100,"t":0,"l":0,"b":0})
fig.show()
fig2 = go.Figure(go.Scattergeo())
fig2.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
fig2.show()
fig3 = go.Figure(go.Scattergeo())
fig3.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
fig3.update_geos(
    resolution=50,
    showcoastlines=True, coastlinecolor="LightSalmon",
    showland=True, landcolor="LemonChiffon",
    showocean=True, oceancolor="LightBlue",
    showlakes=True, lakecolor="Coral",
    showrivers=True, rivercolor="Thistle"
)
fig3.show()
df = pd.read_csv('../input/housing/State_MedianValuePerSqft_AllHomes.csv') #using pandas, read CSV file

#Construct a new figure, fig4, with a Choropleth Map as the data argument in the Figure method. 
fig4 = go.Figure(data=go.Choropleth(
    locations=df['State'], # The state abreviations "AL, FL, IL, etc., provides enough for plotly to fill in a map"
    z = df['1996-05'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', #parameter
    colorscale = 'Blues',#The color scale makes a gradient color of blues representing the values visually
    autocolorscale = False, 
    colorbar_title = "Price per Sq. Ft (USD)",#colorbar_title is the tile for the colorbar on the right
))

fig4.update_layout(
    title_text = 'Housing Price Per Square Foot, May 1996',   #Title the graph as a whole
    geo_scope='usa', # limit map scope to USA
)

fig4.show()
fig5 = go.Figure(data=go.Choropleth(
    locations=df['State'], # The state abreviations "AL, FL, IL, etc., provides enough for plotly to fill in a map"
    z = df['2019-05'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', #parameter
    colorscale = 'tealrose',
    autocolorscale = False,
    colorbar_title = "Price per Sq. Ft (USD)",#colorbar_title is the tile for the colorbar on the right
))

fig5.update_layout(
    title_text = 'Housing Price Per Square Foot, May 2019',   #Title the graph as a whole
    geo_scope='usa', # only show US map
)

fig5.show()
fig5a = go.Figure(data=go.Choropleth(
    locations=df['State'], # The state abreviations "AL, FL, IL, etc., provides enough for plotly to fill in a map"
    z = df['2019-05'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', #parameter
    colorscale = 'greens',
    autocolorscale = False,
    colorbar_title = "Price per Sq. Ft (USD)",#colorbar_title is the tile for the colorbar on the right
))

fig5a.update_layout(
    title_text = 'Housing Price Per Square Foot, May 2019',   #Title the graph as a whole
    geo_scope='usa', # only show US map
)

fig5a.show()
#print(df.head())
y_vals = np.array(np.sum(df.iloc[:, 4:]) / 50 )

arr = np.array(df.columns) #define an array to store the column names, representing the values of the years 
index =  [0, 1, 2, 3] #delete the first four elements since they are not month values
arr =  np.delete(arr, index)  #delete elements

#plotly takes the data (monthly values) and automatically consolidates them into years!
fig6 = go.Figure(data=go.Scatter(x=arr, y= y_vals))

fig6.update_layout(title='Average Home Values Over Time Over All States',
                   xaxis_title='Year',
                   yaxis_title='Selling Price Per Square Foot')

fig6.show()
import plotly.express as px

singleFamilyValues = pd.read_csv('../input/singlefamhome/SingleFamilyHomeSales.csv')
singleFamilyValues_Scatter = pd.read_csv('../input/singlefamhomescatter/SingleFamilyHomeSales_Scatter.csv' )
singleFamilyValues_Scatter.dropna()

cities = singleFamilyValues['City'].tolist()
april96 = singleFamilyValues['1996-04'].tolist()
april2k = singleFamilyValues['2000-04'].tolist()
april04 = singleFamilyValues['2004-04'].tolist()
april08 = singleFamilyValues['2008-04'].tolist()
april12 = singleFamilyValues['2012-04'].tolist()
april16 = singleFamilyValues['2016-04'].tolist()
april19 = singleFamilyValues['2019-04'].tolist()

scfig = px.scatter(singleFamilyValues_Scatter, x="2008-04", y="Income_Range", color="SizeRank")

scfig.update_layout(title='Home Sales by Household Income and City Rank',
                  yaxis_title='National Household Income Ranking',
                  xaxis_title='April 2008 Home Sale Price')
scfig.show()



fig8 = go.Figure()
fig8.add_trace(go.Bar(
    x=cities,
    y=april96,
    name='April 1996',
    marker_color='indianred'
))

fig8.add_trace(go.Bar(
    x=cities,
    y=april2k,
    name='April 2000',
    marker_color='Crimson'
))

fig8.add_trace(go.Bar(
    x=cities,
    y=april04,
    name='April 2004',
    marker_color='lightpink'
))

fig8.add_trace(go.Bar(
    x=cities,
    y=april08,
    name='April 2008',
    marker_color='lightsalmon'
))

fig8.add_trace(go.Bar(
    x=cities,
    y=april12,
    name='April 2012',
    marker_color='darkblue'
))

fig8.add_trace(go.Bar(
    x=cities,
    y=april16,
    name='April 2016',
    marker_color='mediumslateblue'
))

fig8.add_trace(go.Bar(
    x=cities,
    y=april19,
    name='April 2019',
    marker_color='steelblue'
))


fig8.update_layout(barmode='group',
                  bargap=0.15,
                  bargroupgap=0.1,
                  xaxis_tickangle=0,
                  title='Home Sales Over Time',
                  xaxis_tickfont_size=20,
                  yaxis=dict(
                      title='Average Home Sale Price USD',
                      titlefont_size=16,
                      tickfont_size=14))

fig8.show()


clfig = go.Figure()

clfig.add_trace(go.Scatter(x=cities,
                         y=april96,
                         mode='lines+markers',
                         name='Sales 1996',
                         # marker_color='Orange',
                         line=dict(color='indianred', width=4,)))

clfig.add_trace(go.Scatter(x=cities,
                         y=april2k,
                         mode='lines+markers',
                         name='Sales 2000',
                         # marker_color='Orange',
                         line=dict(color='Crimson', width=4,)))

clfig.add_trace(go.Scatter(x=cities,
                         y=april04,
                         mode='lines+markers',
                         name='Sales 2004',
                         # marker_color='Orange',
                         line=dict(color='lightpink', width=4,)))

clfig.add_trace(go.Scatter(x=cities,
                         y=april08,
                         mode='lines+markers',
                         name='Sales 2008',
                         # marker_color='Orange',
                         line=dict(color='lightsalmon', width=4,)))

clfig.add_trace(go.Scatter(x=cities,
                         y=april16,
                         mode='lines+markers',
                         name='Sales 2016',
                         # marker_color='Orange',
                         line=dict(color='darkblue', width=4,)))

clfig.add_trace(go.Scatter(x=cities,
                         y=april19,
                         mode='lines+markers',
                         name='Sales 2000',
                         # marker_color='Orange',
                         line=dict(color='mediumslateblue', width=4,)))


clfig.update_layout(title='Home Sales Over Time',
                  yaxis_title='Average Home Sale Price USD')


clfig.show()
