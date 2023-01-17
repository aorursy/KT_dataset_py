# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
player_data= pd.read_csv("../input/fifa19/data.csv")
player_data.head()
def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

player_data['Value'] = player_data['Value'].apply(value_to_int)
player_data['Wage'] = player_data['Wage'].apply(value_to_int)
player_data= player_data.drop(player_data.columns[[0, 1, 4, 6, 10]], axis=1)
player_data.head()
player_data.describe()
Overall = go.Scatter(
    x=player_data.Name,
    y=player_data['Overall'].where(player_data['Overall'] > 85)
)
Potential = go.Scatter(
    x=player_data.Name,
    y=player_data['Potential'].where(player_data['Potential'] > 85)
   
)
Value = go.Scatter(
    x=player_data.Name,
    y=player_data.Value
   
)
Wages = go.Scatter(
    x=player_data.Name,
    y=player_data.Wage
   
)
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Overall', 'Potential',
                                                          'Value', 'Wage'))

fig.append_trace(Overall, 1, 1)
fig.append_trace(Potential, 1, 2)
fig.append_trace(Value, 2, 1)
fig.append_trace(Wages, 2, 2)

fig['layout'].update(height=600, width=600, title='Players compared' +
                                                  ' on different parameters')

py.iplot(fig, filename='make-subplots-multiple-with-titles')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
plotPerColumnDistribution(player_data, 84, 5)
Agility = go.Scatter(
    x=player_data['Name'],
    y=player_data['Agility'].where(player_data['Agility'] > 85)
)
Balance = go.Scatter(
    x=player_data.Name,
    y=player_data['Balance'].where(player_data['Balance'] > 85)
   
)
Dribbling = go.Scatter(
    x=player_data.Name,
    y=player_data['Dribbling'].where(player_data['Dribbling'] > 85)
   
)
Sprint = go.Scatter(
    x=player_data.Name,
    y=player_data['SprintSpeed'].where(player_data['SprintSpeed'] > 85)
   
)
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Agility', 'Balance',
                                                          'Dribbling', 'Sprint'))

fig.append_trace(Agility, 1, 1)
fig.append_trace(Balance, 1, 2)
fig.append_trace(Dribbling, 2, 1)
fig.append_trace(Sprint, 2, 2)

fig['layout'].update(height=600, width=600, title='Players compared' +
                                                  ' on different parameters whose ratings >85')

py.iplot(fig, filename='make-subplots-multiple-with-titles')
some_clubs = ('Juventus', 'Real Madrid', 'Paris Saint-Germain', 'FC Barcelona', 'FC Bayern München', 'Manchester City', 'Chelsea')
overall_club = player_data.loc[player_data['Club'].isin(some_clubs) & player_data['Overall']] 
import random
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color
data = [go.Bar(
    x = overall_club.Club,
    y = overall_club.Overall,
    marker = dict(color = random_colors(25))
)]
layout = dict(
         title= "Overall Ratings Distribution by Clubs "
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False )
position_count = player_data['Position'].value_counts()
trace = go.Pie(labels=position_count.index, values=position_count.values, hole=0.6,textinfo= "none")
layout = go.Layout(
    title='Percentage of players by position'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
player_features = (
    'Acceleration', 'Aggression', 'Agility', 
    'Balance', 'BallControl', 'Composure', 
    'Crossing', 'Dribbling', 'FKAccuracy', 
    'Finishing', 'GKDiving', 'GKHandling', 
    'GKKicking', 'GKPositioning', 'GKReflexes', 
    'HeadingAccuracy', 'Interceptions', 'Jumping', 
    'LongPassing', 'LongShots', 'Marking', 'Penalties'
)

from math import pi
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in player_data.groupby(player_data['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(10, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1 
trace1 = go.Scatter3d(
    x=player_data['Dribbling'].where(player_data['Dribbling'] > 85),
    y=player_data['SprintSpeed'].where(player_data['SprintSpeed'] > 85),
    z=player_data['Finishing'].where(player_data['Finishing'] > 85),
    text = player_data.Name,
    mode='markers',
    marker=dict(
        size=12,
        color=random_colors(50),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout=go.Layout(width=800, height=800, title = 'Top players through which you can dribble past whole defense ',
              scene = dict(xaxis=dict(title='Dribbling',
                                      titlefont=dict(color='Orange')),
                            yaxis=dict(title='SprintSpeed',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            zaxis=dict(title='Finishing',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            bgcolor = 'rgb(20, 24, 54)'
                           )
             )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')

trace1 = go.Scatter3d(
    x=player_data['Value'].where(player_data['Value'] > 85),
    y=player_data['Composure'].where(player_data['Composure'] > 85),
    z=player_data['Finishing'].where(player_data['Finishing'] > 85),
    text = player_data.Name,
    mode='markers',
    marker=dict(
        size=12,
        color=random_colors(50),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout=go.Layout(width=800, height=800, title = 'Top players market value due to their finishing ',
              scene = dict(xaxis=dict(title='Value',
                                      titlefont=dict(color='Orange')),
                            yaxis=dict(title='Composure',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            zaxis=dict(title='Finishing',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            bgcolor = 'rgb(20, 24, 54)'
                           )
             )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')
trace1 = go.Scatter3d(
    x=player_data['Wage'].where(player_data['Wage'] > 85),
    y=player_data['Potential'].where(player_data['Potential'] > 85),
    z=player_data['Overall'].where(player_data['Overall'] > 85),
    text = player_data.Name,
    mode='markers',
    marker=dict(
        size=12,
        color=random_colors(50),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout=go.Layout(width=800, height=800, title = 'Top players with high market values  ',
              scene = dict(xaxis=dict(title='Value',
                                      titlefont=dict(color='Orange')),
                            yaxis=dict(title='Potential',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            zaxis=dict(title='Overall',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            bgcolor = 'rgb(20, 24, 54)'
                           )
             )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')
trace1 = go.Scatter3d(
    x=player_data['Value'].where(player_data['Value'] > 85),
    y=player_data['GKDiving'].where(player_data['GKDiving'] > 85),
    z=player_data['GKReflexes'].where(player_data['GKReflexes'] > 85),
    text = player_data.Name,
    mode='markers',
    marker=dict(
        size=12,
        color=random_colors(50),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout=go.Layout(width=800, height=800, title = 'Top Goalkeepers who can save goals ',
              scene = dict(xaxis=dict(title='Value',
                                      titlefont=dict(color='Orange')),
                            yaxis=dict(title='GKDiving',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            zaxis=dict(title='GKReflexes',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            bgcolor = 'rgb(20, 24, 54)'
                           )
             )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')
trace1 = go.Scatter3d(
    x = player_data['HeadingAccuracy'].where(player_data['HeadingAccuracy'] > 85),
    y = player_data['Jumping'].where(player_data['Jumping'] > 85),
    z = player_data['Finishing'].where(player_data['Finishing'] > 85),
    text = player_data.Name,
    mode = 'markers',
    marker = dict(
        color = random_colors(50),
        )
)
data=[trace1]

layout=go.Layout(width=800, height=800, title = 'Players through which you can score header goals ',
              scene = dict(xaxis=dict(title='Headingaccuracy',
                                      titlefont=dict(color='Orange')),
                            yaxis=dict(title='Jumping',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            zaxis=dict(title='Finishing',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            bgcolor = 'rgb(20, 24, 54)'
                           )
             )

fig=go.Figure(data=data, layout=layout)
py.iplot(fig, filename='solar_system_planet_size')
trace1 = go.Scatter3d(
    x = player_data['StandingTackle'].where(player_data['StandingTackle'] > 85),
    y = player_data['SlidingTackle'].where(player_data['SlidingTackle'] > 85),
    z = player_data['Strength'].where(player_data['Strength'] > 85),
    text = player_data.Name,
    mode = 'markers',
    marker = dict(
        color = random_colors(50),
        )
)
data=[trace1]

layout=go.Layout(width=800, height=800, title = 'Players through which you can make fine tackle ',
              scene = dict(xaxis=dict(title='StandingTackle',
                                      titlefont=dict(color='Orange')),
                            yaxis=dict(title='SlidingTackle',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            zaxis=dict(title='Strength',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            bgcolor = 'rgb(20, 24, 54)'
                           )
             )

fig=go.Figure(data=data, layout=layout)
py.iplot(fig, filename='solar_system_planet_size')
trace1 = go.Scatter3d(
    x = player_data['StandingTackle'].where(player_data['StandingTackle'] > 85),
    y = player_data['SlidingTackle'].where(player_data['SlidingTackle'] > 85),
    z = player_data['Marking'].where(player_data['Marking'] > 85),
    text = player_data.Name,
    mode = 'markers',
    marker = dict(
        color = random_colors(50),
        )
)
data=[trace1]

layout=go.Layout(width=800, height=800, title = 'Players through which you can make fine tackle ',
              scene = dict(xaxis=dict(title='StandingTackle',
                                      titlefont=dict(color='Orange')),
                            yaxis=dict(title='SlidingTackle',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            zaxis=dict(title='Marking',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                            bgcolor = 'rgb(20, 24, 54)'
                           )
             )

fig=go.Figure(data=data, layout=layout)
py.iplot(fig, filename='solar_system_planet_size')
country_count = player_data.Nationality.value_counts().reset_index()
country_count.columns = ['country', 'players']
locations = go.Bar(x=country_count.values[0:10],y=country_count.index[0:10], marker=dict(color='#CF1020'))
# I use dataset from plotly to get country codes, which are required to plot the data.
country_code = pd.read_csv('../input/plotly-country-code-mapping/2014_world_gdp_with_codes.csv')
country_code.columns = [i.lower() for i in country_code.columns]
country_count.loc[country_count['country'] == 'United States of America', 'country'] = 'United States'
country_count.loc[country_count['country'] == 'United Kingdom of Great Britain and Northern Ireland', 'country'] = 'United Kingdom'
country_count.loc[country_count['country'] == 'South Korea', 'country'] = '"Korea, South"'
country_count.loc[country_count['country'] == 'Viet Nam', 'country'] = 'Vietnam'
country_count.loc[country_count['country'] == 'Iran, Islamic Republic of...', 'country'] = 'Iran'
country_count.loc[country_count['country'] == 'Hong Kong (S.A.R.)', 'country'] = 'Hong Kong'
country_count.loc[country_count['country'] == 'Republic of Korea', 'country'] = '"Korea, North"'
country_count = pd.merge(country_count, country_code, on='country')
data = [ dict(
        type = 'choropleth',
        locations = country_count['code'],
        z = country_count['players'],
        text = country_count['country'],
        colorscale = 'Viridis',
        autocolorscale = True,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(120,120,120)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Players'),
      ) ]

layout = dict(
    title = 'Players by country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
             type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
# Top 5 Vision players
print('Top 5 players with vision: \n',
      player_data[player_data['Vision'] > 90][['Name','Vision']].head())
# Top 5 Vision players
print('Top 5 players with shortpassing ability:\n', player_data[player_data['ShortPassing'] > 90][['Name','ShortPassing']].head())
print('Top 5 players with Dribbling ability:\n', player_data[player_data['Dribbling'] > 90][['Name','Dribbling']].head())
print('Top 5 players with Free kick accuracy:\n',player_data[player_data['FKAccuracy'] > 90][['Name','FKAccuracy']].head())
position_player = player_data.iloc[player_data.groupby(player_data['Position'])['Overall'].idxmax()][['Name', 'Position']]
value_player = player_data.iloc[player_data.groupby(player_data['Value'])['Potential'].idxmax()][['Name', 'Position','Potential','Value']]
position_player
value_player
cheap_players= value_player[value_player['Name'].notnull() & (value_player['Value'] < 30000000.0)]
cheap_players= cheap_players.sort_values(['Potential'], ascending=[False])
pd.set_option('display.max_rows', None)
cheap_players
cheap_players1= value_player[value_player['Name'].notnull() & (value_player['Value'] < 50000000.0)]
cheap_players1= cheap_players1.sort_values(['Potential'], ascending=[False])
cheap_players1
def load_layout():
    """
    Returns a dict for a Football themed Plot.ly layout 
    """
    layout = dict(
        title = "Players Position",
        plot_bgcolor='darkseagreen',
        showlegend=True,
        xaxis=dict(
            autorange=False,
            range=[0, 120],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='',
            tickmode='array',
            tickvals=[10,110],
            ticktext=['Goal', 'Goal'],
            showticklabels=True
        ),
        yaxis=dict(
            title='',
            autorange=False,
            range=[-3.3,56.3],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='',
            showticklabels=False
        ),
        shapes=[
            dict(
                type='line',
                layer='below',
                x0=0,
                y0=0,
                x1=120,
                y1=0,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=0,
                y0=53.3,
                x1=120,
                y1=53.3,
                line=dict(
                    color='white',
                    width=2
                )
            ),
             dict(
                type='line',
                layer='below',
                x0=50,
                y0=0,
                x1=50,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=10,
                y0=0,
                x1=10,
                y1=53.3,
                line=dict(
                    color='white',
                    width=10
                )
            ),
           
            dict(
                type='line',
                layer='below',
                x0=110,
                y0=0,
                x1=110,
                y1=53.3,
                line=dict(
                    color='white',
                    width=10
                )
            )
        ]
    )
    return layout

layout = load_layout()
CB= position_player['Name'].where(position_player['Position'] == 'CB')
GK= position_player['Name'].where(position_player['Position'] == 'GK')
RB= position_player['Name'].where(position_player['Position'] == 'RB')
RCB= position_player['Name'].where(position_player['Position'] == 'RCB')
LCB= position_player['Name'].where(position_player['Position'] == 'LCB')
LB= position_player['Name'].where(position_player['Position'] == 'LB')
CDM= position_player['Name'].where(position_player['Position'] == 'CDM')
LDM= position_player['Name'].where(position_player['Position'] == 'LDM')
RDM= position_player['Name'].where(position_player['Position'] == 'RDM')
LF= position_player['Name'].where(position_player['Position'] == 'LF')
LWB= position_player['Name'].where(position_player['Position'] == 'LWB')
RWB= position_player['Name'].where(position_player['Position'] == 'RWB')
LCM= position_player['Name'].where(position_player['Position'] == 'LCM')
RCM= position_player['Name'].where(position_player['Position'] == 'RCM')
LAM= position_player['Name'].where(position_player['Position'] == 'LAM')
CAM= position_player['Name'].where(position_player['Position'] == 'CAM')
RF= position_player['Name'].where(position_player['Position'] == 'RF')
RS= position_player['Name'].where(position_player['Position'] == 'RS')
RM= position_player['Name'].where(position_player['Position'] == 'RM')
RAM= position_player['Name'].where(position_player['Position'] == 'RAM')
RW= position_player['Name'].where(position_player['Position'] == 'RW')
LS= position_player['Name'].where(position_player['Position'] == 'LS')
LW= position_player['Name'].where(position_player['Position'] == 'LW')
ST= position_player['Name'].where(position_player['Position'] == 'ST')
CB= CB.dropna()
GK= GK.dropna()
RB = RB.dropna()
RCB = RCB.dropna()
LCB = LCB.dropna()
LB = LB.dropna()
CDM = CDM.dropna()
LDM = LDM.dropna()
RDM = RDM.dropna()
LF = LF.dropna()
LWB = LWB.dropna()
RWB = RWB.dropna()
LCM = LCM.dropna()
RCM = RCM.dropna()
LAM = LAM.dropna()
CAM= CAM.dropna()
RF= RF.dropna()
RS= RS.dropna()
RM= RM.dropna()
LS= LS.dropna()
RAM= RAM.dropna()
RW= RW.dropna()
LW= LW.dropna()
ST= ST.dropna()
trace1 = {"x": [12], 
          "y": [25], 
          "marker": {"color": "pink", "size": 12}, 
          "mode": "markers", 
          "text": GK[0:1],
          "name": "GK",
          "type": "scatter"
}
trace2 = {"x": [30], 
          "y": [20], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": LCB[0:1],
          "name": "LCB",
          "type": "scatter"
}
trace3 = {"x": [30], 
          "y": [40], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": RCB[0:1],
          "name": "RCB",
          "type": "scatter"
}
trace4 = {"x": [40], 
          "y": [2], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": RB[0:1],
          "name": "RB",
          "type": "scatter"
}
trace5 = {"x": [40], 
          "y": [48], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": LB[0:1],
          "name": "RB",
          "type": "scatter"
}
trace6 = {"x": [50], 
          "y": [25], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": CDM[0:1],
          "name": "CDM",
          "type": "scatter"
}
trace7 = {"x": [60], 
          "y": [5], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": RCM[0:1],
          "name": "RCM",
          "type": "scatter"
}
trace8 = {"x": [60], 
          "y": [45], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": LCM[0:1],
          "name": "LCM",
          "type": "scatter"
}
trace9 = {"x": [80], 
          "y": [2], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": RF[0:1],
          "name": "RF",
          "type": "scatter"
}
trace10 = {"x": [80], 
          "y": [48], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": LW[0:1],
           "name": "LW",
          "type": "scatter"
}
trace11 = {"x": [90], 
          "y": [25], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": ST[0:1],
           "name": "ST",
          "type": "scatter"
}

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11]

fig=go.Figure(data= data,layout=layout)
py.iplot(fig)
trace1 = {"x": [12], 
          "y": [25], 
          "marker": {"color": "pink", "size": 12}, 
          "mode": "markers", 
          "text": GK[0:1],
          "name": "GK",
          "type": "scatter"
}
trace2 = {"x": [30], 
          "y": [20], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": LCB[0:1],
          "name": "LCB",
          "type": "scatter"
}
trace3 = {"x": [30], 
          "y": [40], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": RCB[0:1],
          "name": "RCB",
          "type": "scatter"
}
trace4 = {"x": [30], 
          "y": [25], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": CB[0:1],
          "name": "CB",
          "type": "scatter"
}
trace5 = {"x": [50], 
          "y": [50], 
          "marker": {"color": "blue", "size": 12}, 
          "mode": "markers", 
          "text": LWB[0:1],
          "name": "LWB",
          "type": "scatter"
}
trace6 = {"x": [60], 
          "y": [25], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": CDM[0:1],
          "name": "CDM",
          "type": "scatter"
}
trace7 = {"x": [50], 
          "y": [2], 
          "marker": {"color": "blue", "size": 12}, 
          "mode": "markers", 
          "text": RWB[0:1],
          "name": "RWB",
          "type": "scatter"
}
trace8 = {"x": [70], 
          "y": [25], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": CAM[0:1],
          "name": "CAM",
          "type": "scatter"
}
trace9 = {"x": [80], 
          "y": [2], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": RF[0:1],
          "name": "RF",
          "type": "scatter"
}
trace10 = {"x": [80], 
          "y": [49], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": LW[0:1],
           "name": "LW",
          "type": "scatter"
}
trace11 = {"x": [90], 
          "y": [25], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": ST[0:1],
           "name": "ST",
          "type": "scatter"
}

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11]

fig=go.Figure(data= data,layout=layout)
py.iplot(fig)
trace1 = {"x": [12], 
          "y": [25], 
          "marker": {"color": "pink", "size": 12}, 
          "mode": "markers", 
          "text": GK[0:1],
          "name": "GK",
          "type": "scatter"
}
trace2 = {"x": [30], 
          "y": [20], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": LCB[0:1],
          "name": "LCB",
          "type": "scatter"
}
trace3 = {"x": [30], 
          "y": [40], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": RCB[0:1],
          "name": "RCB",
          "type": "scatter"
}
trace4 = {"x": [30], 
          "y": [25], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": CB[0:1],
          "name": "CB",
          "type": "scatter"
}
trace5 = {"x": [50], 
          "y": [50], 
          "marker": {"color": "blue", "size": 12}, 
          "mode": "markers", 
          "text": LWB[0:1],
          "name": "LWB",
          "type": "scatter"
}
trace6 = {"x": [60], 
          "y": [30], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": RDM[0:1],
          "name": "RDM",
          "type": "scatter"
}
trace7 = {"x": [50], 
          "y": [2], 
          "marker": {"color": "blue", "size": 12}, 
          "mode": "markers", 
          "text": RWB[0:1],
          "name": "RWB",
          "type": "scatter"
}
trace8 = {"x": [70], 
          "y": [25], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": CAM[0:1],
          "name": "CAM",
          "type": "scatter"
}
trace9 = {"x": [90], 
          "y": [20], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": RS[0:1],
          "name": "RS",
          "type": "scatter"
}
trace10 = {"x": [90], 
          "y": [30], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": LS[0:1],
           "name": "LS",
          "type": "scatter"
}
trace11 = {"x": [60], 
          "y": [20], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": LDM[0:1],
           "name": "LDM",
          "type": "scatter"
}

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11]

fig=go.Figure(data= data,layout=layout)
py.iplot(fig)
trace1 = {"x": [12], 
          "y": [25], 
          "marker": {"color": "pink", "size": 12}, 
          "mode": "markers", 
          "text": GK[0:1],
          "name": "GK",
          "type": "scatter"
}
trace2 = {"x": [30], 
          "y": [20], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": LCB[0:1],
          "name": "LCB",
          "type": "scatter"
}
trace3 = {"x": [30], 
          "y": [40], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": RCB[0:1],
          "name": "RCB",
          "type": "scatter"
}
trace4 = {"x": [30], 
          "y": [25], 
          "marker": {"color": "red", "size": 12}, 
          "mode": "markers", 
          "text": CB[0:1],
          "name": "CB",
          "type": "scatter"
}
trace5 = {"x": [50], 
          "y": [50], 
          "marker": {"color": "blue", "size": 12}, 
          "mode": "markers", 
          "text": LAM[0:1],
          "name": "LAM",
          "type": "scatter"
}
trace6 = {"x": [60], 
          "y": [30], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": RDM[0:1],
          "name": "RDM",
          "type": "scatter"
}
trace7 = {"x": [50], 
          "y": [2], 
          "marker": {"color": "blue", "size": 12}, 
          "mode": "markers", 
          "text": RAM[0:1],
          "name": "RAM",
          "type": "scatter"
}
trace8 = {"x": [70], 
          "y": [25], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": CAM[0:1],
          "name": "CAM",
          "type": "scatter"
}
trace9 = {"x": [90], 
          "y": [20], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": RS[0:1],
          "name": "RS",
          "type": "scatter"
}
trace10 = {"x": [90], 
          "y": [30], 
          "marker": {"color": "green", "size": 12}, 
          "mode": "markers", 
          "text": LS[0:1],
           "name": "LS",
          "type": "scatter"
}
trace11 = {"x": [60], 
          "y": [20], 
          "marker": {"color": "yellow", "size": 12}, 
          "mode": "markers", 
          "text": LDM[0:1],
           "name": "LDM",
          "type": "scatter"
}

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11]

fig=go.Figure(data= data,layout=layout)
py.iplot(fig)