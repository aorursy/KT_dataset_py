import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input/"))

athletes = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
scores = pd.read_csv("../input/divingscores2016/swimscores2016.csv")
athletes.info()
weight = {'Weight'}
improve = athletes.dropna(subset=weight)
#dropping rows which don't have weight values
improve['Height'].isnull().describe()
#looks like about 2000 of these entries don't have height values
height = {'Height'}
improve = improve.dropna(subset=height)
improve.info()
#now we have our data
#We can now select an event to look into first. I'm going to look at Olympic Synchronized Divers first.
#Have to look through the list of events, there are 590:
len(improve['Event'].unique())
improve['Event'].unique()
divingWS = improve.loc[improve['Event'] == "Diving Women's Synchronized Springboard"]
divingWS = divingWS.sort_values(by=['Year'])
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
k = sns.lmplot(data=divingWS,x='Year', y='Weight', hue='Medal', fit_reg=False)
k = sns.lmplot(data=divingWS,x='Year', y='Height', hue='Medal', fit_reg=False)
#The heights and especially the weights are quite close for the Gold winners.
#Going to add in the non-medalists to see how they differ
wdiving = divingWS
wdiving['Medal'] = wdiving['Medal'].fillna('Non-Medalists')
#filling the nulls of the Medal's column
k = sns.lmplot(data=divingWS,x='Year', y='Weight', hue='Medal', fit_reg=False)
k = sns.lmplot(data=divingWS,x='Year', y='Height', hue='Medal', fit_reg=False)

k = sns.lmplot(data=wdiving,x='Year', y='Height', hue = 'Team',markers=['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', '|', 'H','>','<',','], fit_reg=False )
#markers=['o','x','^','+']


#We can see that there are many countries, hard to get any information out of this
#focus it down to a specific year: 2016
k = sns.lmplot(data=wdiving[wdiving['Year'] == 2016],x='Team', y='Height', hue = 'Medal', fit_reg=False )
k = sns.lmplot(data=wdiving[wdiving['Year'] == 2016],x='Team', y='Weight', hue = 'Medal', fit_reg=False )
#We can see that the heights are always quite close, usually a difference of a couple centimeters
#Weights can vary, I don't believe that there is a huge effect from either.
g = sns.FacetGrid(data=wdiving,col = 'Year', row = 'Medal', hue = 'Team' )
g = g.map(plt.scatter,'Height', 'Weight' )
plt.legend()
team = wdiving[wdiving['Year'] == 2016]
heightweightteam = ['Height', 'Weight', 'Team']
team = team[heightweightteam]
team.reset_index()
team
team = team.sort_values('Team').reset_index()
team = team.drop('index', axis=1)


team['diffheight'] = abs(team['Height'].diff())
team['diffweight'] = abs(team['Weight'].diff())
team['distance'] = (team['diffheight'])**2 + (team['diffweight'])**2
team['distance'] = np.sqrt(team['distance'])
team = team[1::2]

team = team.drop(['Height', 'Weight'], axis=1)


team.columns = ['team','diffheight', 'diffweight','distance']
team
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

trace1 = go.Bar(
    x=scores.team,
    y=scores.dive1,
    name='Dive 1'
)
trace2 = go.Bar(
    x=scores.team,
    y=scores.dive2,
    name='Dive 2'
)
trace3 = go.Bar(
    x=scores.team,
    y=scores.dive3,
    name='Dive 3'
)
trace4 = go.Bar(
    x=scores.team,
    y=scores.dive4,
    name='Dive 4'
)
trace5 = go.Bar(
    x=scores.team,
    y=scores.dive5,
    name='Dive 5'
)

data = [trace1, trace2,trace3,trace4,trace5]
layout = go.Layout(
    barmode='group'
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
mix = scores.merge(team, how='left',on='team')
mix
mix.columns = ['scrap','team','dive1','dive2','dive3','dive4','dive5','total','distance','diffheight','diffweight']
mix = mix.drop('scrap', axis=1)
corr = mix.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})