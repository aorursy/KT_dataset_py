# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import altair as alt

from collections import Counter, OrderedDict
from IPython.display import HTML
from altair.vega import v3
#alt.renderers.enable('notebook')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
##-----------------------------------------------------------
# This whole section 
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

#------------------------------------------------ Defs for future rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>",
    "This code block sets up embedded rendering in HTML output and<br/>",
    "provides the function `render(chart, id='vega-chart')` for use below."
)))
# Create matches and deliveries DataFrames
matches = pd.read_csv("../input/matches.csv")
deliveries = pd.read_csv("../input/deliveries.csv")
matches.head()
deliveries.head()
# Check the number of rows and columns in matches dataset
matches.shape
# Check the number of rows and columns in deliveries dataset
deliveries.shape
# Check if there are any null value columns.
matches.isnull().sum()
# Drop Umpire3 column from matches dataframe.
matches.drop(['umpire3'],axis=1,inplace=True)
# Validate to see the umpire3 column has been dropped.

matches.columns
# Teams played in IPL from 2008 to 2017

np.unique(np.concatenate((matches['team1'].unique(), matches['team2'].unique()), axis=0))
# How many unique teams have played IPL till now?

print("There are {} teams played in IPL till 2017 season".format(len(np.unique(np.concatenate((matches['team1'].unique(), matches['team2'].unique()), axis=0)))))
# What are the seasons available to us?
np.sort(matches['season'].unique())
# Modify the team names in matches DataFrame to their short forms.

matches.replace(['Chennai Super Kings', 'Deccan Chargers', 'Delhi Daredevils',
       'Gujarat Lions', 'Kings XI Punjab', 'Kochi Tuskers Kerala',
       'Kolkata Knight Riders', 'Mumbai Indians', 'Pune Warriors',
       'Rajasthan Royals', 'Rising Pune Supergiant',
       'Rising Pune Supergiants', 'Royal Challengers Bangalore',
       'Sunrisers Hyderabad'],
               ['CSK', 'DC', 'DD', 'GL', 'KXIP', 'KTK', 'KKR', 'MI', 'PW', 'RR', 'RPS', 'RPS', 'RCB', 'SRH'],inplace=True)

# Modify the team names in deliveries DataFrame to their short forms.

deliveries.replace(['Chennai Super Kings', 'Deccan Chargers', 'Delhi Daredevils',
       'Gujarat Lions', 'Kings XI Punjab', 'Kochi Tuskers Kerala',
       'Kolkata Knight Riders', 'Mumbai Indians', 'Pune Warriors',
       'Rajasthan Royals', 'Rising Pune Supergiant',
       'Rising Pune Supergiants', 'Royal Challengers Bangalore',
       'Sunrisers Hyderabad'],
               ['CSK', 'DC', 'DD', 'GL', 'KXIP', 'KTK', 'KKR', 'MI', 'PW', 'RR', 'RPS', 'RPS', 'RCB', 'SRH'],inplace=True)
# Total number of matches played in IPL till end of 2017 season.

print("Total Matches played: ", matches.shape[0])
# Cities at which the IPL matches were played.

print(" \n Cities Played at: ", matches['city'].dropna().unique())
# Unique teams played in IPL

print(" \n Unique teams played: ",np.unique(np.concatenate((matches['team1'].unique(), matches['team2'].unique()), axis=0)))
# Number of Unique teams played in IPL
print(" \n Number of Unique teams played: ", len(matches['team1'].unique()))
# Venues where the IPL MATCHES were played 
    
print ("Venues for IPL matches:\n")
for i in matches['venue'].unique():
    print(i)
# Number of unique venues
print("Number of Unique Venues used:", matches['venue'].nunique())
# Number of Unique cities 
print("Number of Unique cities played at: ", matches["city"].nunique())
#  Which team has won the toss most times.
matches['toss_winner'].value_counts()
matches['toss_winner'].value_counts().idxmax()
# Which team has won the match most times.

most_match_winnning_team = matches['winner'].value_counts().idxmax()
print("{} has won the matches most number of times.".format(most_match_winnning_team))
# Which city has hosted most IPL matches

most_matches_hosting_city = matches['city'].value_counts().idxmax()
print("{} has hosted most IPL matches.".format(most_matches_hosting_city))
# Who has won player of the match most?

most_times_player = matches['player_of_match'].value_counts().idxmax()
print("'{}' was the player of match most times across IPL Seasons".format(most_times_player))
# Finding Maximum win_by_runs from all the seasons.

matches['win_by_runs'].max()
# What were the teams participated in the 146 win by runs match.

wbr = matches.iloc[[matches['win_by_runs'].idxmax()]]
wbr[['season', 'team1', 'team2', 'toss_winner', 'winner','win_by_wickets', 'win_by_runs']]
# Finding maximum win_by_wickets from all seasons.
# Showing only few columns for that row.

wbw = matches.iloc[[matches['win_by_wickets'].idxmax()]]
wbw[['season', 'team1', 'team2', 'toss_winner', 'winner', 'win_by_wickets', 'win_by_runs']]

chart = alt.Chart(matches).mark_bar().encode(
# alt.X('season:N', scale=alt.Scale(rangeStep=50), axis=alt.Axis(title='IPL Seasons')),
    alt.X('season:N', scale=alt.Scale(rangeStep=50), axis=alt.Axis(title='IPL Seasons')),
    alt.Y('count(id)', axis=alt.Axis(title='Number of matches')),
    color=alt.Color('season', scale=alt.Scale(scheme='plasma'), legend=None),
    tooltip=['season', alt.Tooltip('count(id)', title='Number of Matches')]
#color=alt.condition(if_false=)
).configure_axis(
    domainWidth=1,
    titleFontSize=20,
    labelFontSize=15
).properties(
    height=450,
    width=600,
    title='Number of Matches played in each season')

render(chart, id='vega-chart')
# Toss Decisions across Seasons
chart = alt.Chart(matches).mark_bar().encode(
    alt.X('toss_decision', axis=alt.Axis(title='')),
    alt.Y('count(id)', axis=alt.Axis(title='Number of decisions')),
    column='season',
    color=alt.Color('toss_decision',title='Toss Decision'),
    tooltip=['season', alt.Tooltip('count(id)', title='Count')]
).configure_axis(
    domainWidth=1,
    titleFontSize=20,
    labelFontSize=15
).properties(height=400, width=50).configure_view(
    stroke='transparent'
)
render(chart, id='vega-chart')
# Validate above result by finding the number of "field" and "bat" toss_decisions 
matches.loc[(matches['season']==2008) & (matches['toss_decision']=='field')].shape
matches.loc[(matches['season']==2008) & (matches['toss_decision']=='bat')].shape
# Number of times each team has won Toss

matches['toss_winner'].value_counts()


# Maximum Times Toss winner across seasons
chart=alt.Chart(matches).mark_bar().encode(
    alt.X('toss_winner', sort=alt.EncodingSortField(field='count():Q', op='count', order='descending'), axis=alt.Axis(title='Toss Winning Team')),
#     x = "np.sort(matches['toss_winner'].value_counts()):N",
    alt.Y('count()', axis=alt.Axis(title='Number of Toss Wins')),
#     color = 'toss_winner',
    color = alt.Color('toss_winner', title='Toss Winner', scale=alt.Scale(scheme='viridis')),
    order = alt.Order( 'count()', sort = 'ascending'),
    tooltip = [alt.Tooltip('toss_winner', title='Toss Winner'), alt.Tooltip('count(id)', title='count')]
).configure_axis(
    domainWidth=1,
    labelFontSize=15,
    titleFontSize=20
).properties(height=400, width=600, title='Toss Winners Across Seasons')
render(chart, id='vega-chart')
mpt = pd.concat([matches['team1'], matches['team2']])
mpt = mpt.value_counts().reset_index()
mpt.columns = ['Team', 'Number_of_matches_played']
mpt['wins']=matches['winner'].value_counts().reset_index()['winner']

total_matches_played = alt.Chart(mpt).mark_bar().encode(
    alt.X('Team',axis=alt.Axis(title='Teams Played in IPL'), sort=alt.EncodingSortField(field='Number_of_matches_played:Q', op='count', order='ascending')),  
    alt.Y('Number_of_matches_played:Q', axis=alt.Axis(title='Matches Played') ),
    tooltip=['Team', alt.Tooltip('sum(Number_of_matches_played)', title='Total Matches Played')],
).properties(height=400, width=600)

wins = alt.Chart(mpt).mark_bar(color='orange').encode(
    alt.X('Team',axis=alt.Axis(title='Teams Played in IPL'), sort=alt.EncodingSortField(field='Number_of_matches_played:Q', op='count', order='ascending')), 
    alt.Y('wins', axis=alt.Axis(title='Wins')), 
    tooltip=['Team', alt.Tooltip('sum(wins)', title='Matches Won')],
).properties(height=400, width=600)

chart = alt.layer(total_matches_played, wins, title='Total Matches Played vs Total Wins')
render(chart, id='vega-chart')
deliveries_by_season = matches[['id', 'season']].merge(deliveries, left_on='id', right_on='match_id', how='left').drop('id', axis=1)
runsBySeason = deliveries_by_season.groupby(['season'])['total_runs'].sum().reset_index()
# Runs across the Seasons
chart=alt.Chart(runsBySeason).mark_line(point=True).encode(
    alt.X('season:N', axis=alt.Axis(title='IPL Seasons')),
    alt.Y('total_runs:Q', scale=alt.Scale(zero=False),axis=alt.Axis(title='Total Runs')),
    tooltip=['season', alt.Tooltip('total_runs', title='Runs Scored')]
    ).configure_axis(
    domainWidth=1,
    labelFontSize=15,
    titleFontSize=20
).properties(height=400, width=600, title='Total Runs scored in each Season')
render(chart, id='vega-chart')
avgRuns = matches.groupby(['season']).count().id.reset_index()
avgRuns.rename(columns={'id':'num_matches'}, inplace=True)
avgRuns['total_runs'] = runsBySeason['total_runs']
avgRuns['avg_runs_this_season']=avgRuns['total_runs']/avgRuns['num_matches']

chart=alt.Chart(avgRuns).mark_line(point=True).encode(
    alt.X('season:N', scale=alt.Scale(zero=False), axis=alt.Axis(title='IPL Seasons')),
    alt.Y('avg_runs_this_season:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='Average runs')),
    tooltip=['season', alt.Tooltip('avg_runs_this_season', title='Average Runs')]
    ).configure_axis(
        domainWidth=1,
        labelFontSize=15,
        titleFontSize=20
        ).properties(height=400, width=600, title='Average Runs scored across seasons')
render(chart, id='vega-chart')
# Below command is used to bypass the error
#(MaxRowsError: The number of rows in your dataset is greater than the maximum allowed (5000). 
# For information on how to plot larger datasets in Altair, see the documentation)

#alt.data_transformers.enable('json')
alt.data_transformers.enable('default', max_rows=None)
# Sixes and Fours Across the Season

boundaries = deliveries_by_season[(deliveries_by_season['batsman_runs'] == 4) | (deliveries_by_season['batsman_runs'] == 6)]

chart=alt.Chart(boundaries).mark_line(point=True).encode(
    alt.X('season:N', axis=alt.Axis(title='IPL Seasons')),
    alt.Y('count()', axis=alt.Axis(title='Number of Boundaries')),
    tooltip=(alt.Tooltip('season',title='season'), alt.Tooltip('count()', title='boundaries')),
    color=alt.Color('batsman_runs:O', legend=alt.Legend(title='Boundaries'), scale=alt.Scale(domain=[4,6], range=['green', 'blue'])),    
).configure_axis(
    domainWidth=1,
    labelFontSize=15,
    titleFontSize=20
).properties(width=600, height=400, title='Boundaries across IPL Seasons')

render(chart, id='vega-chart')

# Favorite Grounds
chart=alt.Chart(matches).mark_bar().encode(
    alt.Y('venue:O', sort=alt.EncodingSortField(field='count():Q', op='count', order='descending')),
    alt.X('count():Q'),
    color=alt.Color('count():Q', scale=alt.Scale(scheme='purpleorange'), legend=None),
    tooltip=[ 'venue', alt.Tooltip('count()', title='number of matches'), 'city']
    ).configure_axis(
    domainWidth=5,
    labelFontSize=15,
    titleFontSize=20
    ).properties(height=700, width=600, title='Favourite Grounds for IPL')
render(chart, id='vega-chart')
# Maximum Man Of Matches

chart=alt.Chart(matches).mark_bar().encode(
    alt.Y('count():Q', axis=alt.Axis(title='Number of Titles')),
    alt.X('player_of_match:O',  axis=alt.Axis(title='Man of the Match'), sort=alt.EncodingSortField(field='count():Q', op='count', order='descending' )),
    alt.Color('count():Q', scale=alt.Scale(scheme='viridis'), legend=None),
    tooltip=(alt.Tooltip('player_of_match:O', title='player'), alt.Tooltip('count():Q', title='number of titles')),
    ).configure_axis(
        domainWidth=1,
        titleFontSize=20
    ).properties(height=700, width=2500, title='Man of the match')
render(chart, id='vega-chart')
top_20_players = matches['player_of_match'].value_counts().head(20).reset_index()

top_20_players.columns=('Player', 'Number_of_Titles')
# top 20 "man of the match" winners

chart=alt.Chart(top_20_players).mark_bar().encode(
    alt.X('Number_of_Titles:Q', axis=alt.Axis(title='Number of Awards'), ),
    alt.Y('Player:O', axis=alt.Axis(title='Man of the Match'), sort=alt.EncodingSortField(field='Number_of_Titles', op='count')),
    alt.Color('Number_of_Titles:Q', scale=alt.Scale(scheme='purpleorange'), legend=None),
    tooltip=['Player:O', alt.Tooltip('Number_of_Titles:Q', title='Number of awards')]
    ).configure_axis(
        domainWidth=1,
        labelFontSize=15,
        titleFontSize=20).properties(height=600, width=600)
render(chart, id='vega-chart')