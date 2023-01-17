import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 
import pandas as pd
df = pd.read_csv('../input/salaries/2014_World_Power_Consumption')
df.head()
data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        reversescale = True,
        locations = df['Country'],
        locationmode = "country names",
        z = df['Power Consumption KWH'],
        text = df['Country'],
        colorbar = {'title' : 'Power Consumption KWH'},
      ) 

layout = dict(title = '2014 Power Consumption KWH',
              geo=dict(
        showframe=False,
#         showcoastlines=False,
        projection_type='equirectangular'
    )
#                 geo = dict(showframe = False,projection = {'type':'Mercator'})
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
df2 = pd.read_csv("../input/salaries/2012_Election_Data")
df2.head()
data = dict(type='choropleth',
            colorscale = 'Viridis',
            reversescale = True,
            locations = df2['State Abv'],
            z = df2['Voting-Age Population (VAP)'],
            locationmode = 'USA-states',
            text = df2['State'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),
            colorbar = {'title':"Voting-Age Population (VAP)"}
            ) 
layout = dict(title = '2012 General Election Voting Data',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)