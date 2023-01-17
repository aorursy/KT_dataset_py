import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn as sk
import plotly
%matplotlib inline
# import data
df1 = pd.read_csv('../input/athlete_events.csv')
df2 = pd.read_csv('../input/noc_regions.csv')
df1.head()
df2.head()
df1.info()
df2.info()
event_weights = pd.DataFrame(df1.groupby('Event', as_index = False)['Weight'].mean())
event_weights.head()
df1.Weight = df1.Weight.mask(df1.Weight.eq(0)).fillna(
    df1.Event.map(event_weights.set_index('Event').Weight))
df1.info()
event_heights = pd.DataFrame(df1.groupby('Event', as_index = False)['Height'].mean())
df1.Height = df1.Height.mask(df1.Height.eq(0)).fillna(
    df1.Event.map(event_heights.set_index('Event').Height))
event_ages = pd.DataFrame(df1.groupby('Event', as_index = False)['Age'].mean())
df1.Age = df1.Age.mask(df1.Age.eq(0)).fillna(
    df1.Event.map(event_ages.set_index('Event').Age))
men_weight = df1['Weight'].loc[df1['Sex']=='M'].mean()
women_weight = df1['Weight'].loc[df1['Sex']=='F'].mean()
men_height = df1['Height'].loc[df1['Sex']=='M'].mean()
women_height = df1['Height'].loc[df1['Sex']=='F'].mean()
men_age = df1['Age'].loc[df1['Sex']=='M'].mean()
women_age = df1['Age'].loc[df1['Sex']=='F'].mean()
df1['Weight'].loc[df1['Sex']=='M'] = df1['Weight'].loc[df1['Sex']=='M'].fillna(men_weight)
df1['Weight'].loc[df1['Sex']=='F'] = df1['Weight'].loc[df1['Sex']=='F'].fillna(women_weight)
df1['Height'].loc[df1['Sex']=='M'] = df1['Height'].loc[df1['Sex']=='M'].fillna(men_height)
df1['Height'].loc[df1['Sex']=='F'] = df1['Height'].loc[df1['Sex']=='F'].fillna(women_height)
df1['Age'].loc[df1['Sex']=='M'] = df1['Age'].loc[df1['Sex']=='M'].fillna(men_age)
df1['Age'].loc[df1['Sex']=='F'] = df1['Age'].loc[df1['Sex']=='F'].fillna(women_age)
df1.info()
df1['Medal'] = df1['Medal'].fillna('None')
df1.info()
data_final = df1.merge(df2, left_on = 'NOC', right_on='NOC', how = 'left')
data_final.head()
data_final['notes'] = data_final['notes'].fillna('None')
data_final['region'] = data_final['region'].fillna(data_final['Team'])
data_final.info()
data_final['Medal'] = data_final['Medal'].astype(str)
winners = data_final.loc[data_final['Medal'] != 'None']
winners.head()
medals = winners.pivot_table('Medal', ['region','Year'],aggfunc='count')
print(medals.info())
medals = medals.reset_index()
medals.head()
medals['region'] = medals['region'].astype(str)
medals['Year'] = pd.to_datetime(medals['Year'], format='%Y')
reg_medals = medals.groupby('region', as_index=False)['Medal'].sum()
reg_medals = reg_medals.sort_values(['Medal']).reset_index(drop=True)
plt.figure(figsize=(20, 10))
ax = sns.barplot(reg_medals['region'],reg_medals['Medal'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.title('Frequency of Medalists by Country')
plt.show()
# setting date as index 
medals.set_index('Year',inplace=True)
# bring in plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# set up different countries
uk = medals.loc[medals['region']=='UK']
germany = medals.loc[medals['region']=='Germany']
canada = medals.loc[medals['region']=='Canada']
usa = medals.loc[medals['region']=='USA']
russia = medals.loc[medals['region']=='Russia']

#select data
count_uk = uk['Medal']
year_uk = uk.index

count_germany = germany['Medal']
year_germany = germany.index

count_canada = canada['Medal']
year_canada = canada.index

count_usa = usa['Medal']
year_usa = usa.index

count_russia = russia['Medal']
year_russia = russia.index

#create traces
trace_uk = go.Scatter(
    x=year_uk,
    y=count_uk,
    name = "United Kingdom",
    line = dict(color = 'rgb(244,66,66)'),
    opacity = 0.8)

trace_germany = go.Scatter(
    x=year_germany,
    y=count_germany,
    name = "Germany",
    line = dict(color = 'rgb(244,232,66)'),
    opacity = 0.8)
    
trace_canada = go.Scatter(
    x=year_canada,
    y=count_canada,
    name = "Canada",
    line = dict(color = 'rgb(89,244,66)'),
    opacity = 0.8)

trace_usa = go.Scatter(
    x=year_usa,
    y=count_usa,
    name = "USA",
    line = dict(color = 'rgb(244,173,66)'),
    opacity = 0.8)
    
trace_russia = go.Scatter(
    x=year_russia,
    y=count_russia,
    name = "Russia",
    line = dict(color = 'rgb(191,66,244)'),
    opacity = 0.8)

data = [trace_uk,trace_germany,trace_canada,trace_usa,trace_russia]

layout = dict(
    title='Olympic Medalists from around the World over 120 years',
    xaxis=dict(rangeslider=dict(visible = True),type='date'))
fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig)
total_medals = pd.DataFrame(winners.groupby(['NOC','region'])['Medal'].count())
print(total_medals.info())
total_medals = total_medals.reset_index()
data = [ dict(
        type = 'choropleth',
        locations = total_medals['NOC'],
        z = total_medals['Medal'],
        text = total_medals['region'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Olympic Medalists'),
      ) ]

layout = dict(
    title = 'Olympic Medalists by Country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plotly.offline.iplot( fig, validate=False, filename='d3-world-map' )
# group for weights and heights
height_weight_avg = data_final.groupby(['region','Sex'], as_index=False)['Height','Weight'].mean()
# Create men/women frames
men=height_weight_avg.loc[height_weight_avg['Sex']=='M']
women=height_weight_avg.loc[height_weight_avg['Sex']=='F']
# create traces
trace0 = go.Scatter(
    x = men['Height'],
    y = men['Weight'],
    name = 'Male',
    mode = 'markers',
    text = men['region'],
    marker = dict(
        size = 10,
        color = 'rgba(255, 0, 0, .8)',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)',
        )
    )
)

trace1 = go.Scatter(
    x = women['Height'],
    y = women['Weight'],
    name = 'Female',
    mode = 'markers',
    text = women['region'],
    marker = dict(
        size = 10,
        color = 'rgba(34, 19, 242, .9)',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)',
        )
    )
)

data = [trace0, trace1]

layout = dict(title = 'Who has the biggest Athletes?',
              hovermode= 'closest',
              xaxis= dict(
                title= 'Height',
                ticklen= 5,
                zeroline= False,
                gridwidth= 2,
              ),
              yaxis=dict(
                title= 'Weight',
                ticklen= 5,
                gridwidth= 2,
              ),
             )

fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig, filename='styled-scatter')