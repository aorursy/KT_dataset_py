import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px
df_as = pd.read_csv('../input/nba-all-star-game-20002016/NBA All Stars 2000-2016 - Sheet1.csv')
df_as.head()
df_as['Western'] = 0

df_as['Eastern'] = 0

df_as['Fan Vote'] = 0

df_as['Coaches'] = 0

df_as['Replacement'] = 0
for idx, sel in enumerate(df_as['Selection Type']):

    if 'Western' in sel:

        df_as.loc[idx, 'Western'] = 1

    else :

        df_as.loc[idx, 'Eastern'] = 1

    if 'Fan Vote' in sel:

        df_as.loc[idx, 'Fan Vote'] = 1

    elif 'Coaches' in sel:

        df_as.loc[idx, 'Coaches'] = 1

    else :

        df_as.loc[idx, 'Replacement'] = 1
def get_draft_year(s):

    return int(s[0:4])



df_as['Draft Year'] = df_as['NBA Draft Status'].apply(get_draft_year)
def get_draft_order(s):

    l = s.split()

    if 'Undrafted' in s:

        return None

    elif l[2] == '1':

        return l[4]

    else :

        return int(l[4]) + 30

    

df_as['Overall Draft Order'] = df_as['NBA Draft Status'].apply(get_draft_order)
for idx, nat in enumerate(df_as['Nationality']):

    if '\n' in nat:

        l = nat.split('\n')

        df_as.loc[idx, 'First Nationality'] = l[0]

        df_as.loc[idx, 'Second Nationality'] = l[1]

    else :

        df_as.loc[idx, 'First Nationality'] = nat

        df_as.loc[idx, 'Second Nationality'] = None
df_as.head()
nat_counts = df_as[['First Nationality', 'Second Nationality']].apply(pd.value_counts)

labels = nat_counts.index

nat_counts = nat_counts.fillna(0)

values = nat_counts['First Nationality'] + nat_counts['Second Nationality']



fig = px.pie(df_as, values=values, names=labels, title='All-Stars Nationality Distribution')

fig.update_layout(    margin=dict(

        l=50,

        r=50,

        b=100,

        t=200,

        pad=4

    ))

fig.show()
labels = ['Western', 'Eastern']

values = list([sum(df_as['Western']), sum(df_as['Eastern'])])

fig = px.pie(df_as, values=values, names=labels, title='All-Stars West and East Distribution')

fig.show()
labels = ['Fan Vote', 'Coaches', 'Replacement']

values = list([sum(df_as['Fan Vote']), sum(df_as['Coaches']), sum(df_as['Replacement'])])

fig = px.pie(df_as, values=values, names=labels, title='All-Stars Selection Distribution')

fig.show()
labels = df_as['Team'].value_counts().index

values = df_as['Team'].value_counts().values

fig = px.pie(df_as, values=values, names=labels, title='All-Stars Team Distribution')

fig.show()
labels = df_as['Player'].value_counts().index

values = df_as['Player'].value_counts().values

fig = px.bar(df_as, y=values, x=labels, title='All-Stars Players Distribution')

fig.update_layout(

    xaxis_title = 'Name of Players',

    yaxis_title = 'Number of All-Star Appearances'

)

fig.show()
df_as.loc[df_as['Pos'] == 'F-C', 'Pos'] = 'FC'

df_as.loc[df_as['Pos'] == 'G-F', 'Pos'] = 'GF'
labels = df_as['Pos'].value_counts().index

values = df_as['Pos'].value_counts().values

fig = px.pie(df_as, values=values, names=labels, title='All-Stars Position Distribution')

fig.show()
import re

r = re.compile(r"([0-9]+)-([0-9]*[0-9]+)")

def get_inches(el):

    m = r.match(el)

    if m == None:

        return float('NaN')

    else:

        return int(m.group(1))*12 + float(m.group(2))

df_as['HT'] = df_as['HT'].apply(get_inches)
fig = px.box(df_as, x = 'Pos', y = 'HT', title = 'All-Stars Height per Position')

fig.show()
fig = px.box(df_as, x = 'Pos', y = 'WT', title = 'All-Stars Weight per Position')

fig.show()
players = df_as[['Player', 'Draft Year']]

players = players.drop_duplicates()

labels = players['Draft Year'].value_counts().index

values = players['Draft Year'].value_counts().values

fig = px.bar(players, x = labels, y = values, title='All-Stars Draft Year Distribution')

fig.update_layout(

    xaxis_title = 'Year',

    yaxis_title = 'Number of All-Stars'

)

fig.show()
players = df_as[['Player', 'Draft Year', 'Overall Draft Order']]

players = players.drop_duplicates()

labels = players['Overall Draft Order'].value_counts().index

values = players['Overall Draft Order'].value_counts().values

fig = px.bar(players, x = labels, y = values, title='All-Stars Overall Draft Order Distribution')

fig.update_layout(

    xaxis_title = 'Draft Order',

    yaxis_title = 'Number of All-Stars Selected'

)

fig.show()
df_stats = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')

df_stats.head()
df_stats.columns
df_stats['TRB'] = df_stats['TRB'] / df_stats['G']

df_stats['AST'] = df_stats['AST'] / df_stats['G']

df_stats['STL'] = df_stats['STL'] / df_stats['G']

df_stats['BLK'] = df_stats['BLK'] / df_stats['G']

df_stats['PTS'] = df_stats['PTS'] / df_stats['G']



chosen_features = ['Player', 'Year', 'Age', 'PER', 'TS%', 'VORP', 'FG%', '3P%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']

df_stats = df_stats[chosen_features]
df_stats.head()
df_combined = pd.merge(df_as, df_stats, on = ['Player', 'Year'])
df_combined.head()
labels = df_combined['Age'].value_counts().index

values = df_combined['Age'].value_counts().values

fig = px.bar(df_combined, x = labels, y = values, title='All-Stars Age Distribution')

fig.update_layout(

    xaxis_title = 'Age',

    yaxis_title = 'Number of All-Stars Selected'

)

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'PER', color = 'Pos', title='All-Stars PER Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'VORP', color = 'Pos', title='All-Stars VORP Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'TS%', color = 'Pos', title='All-Stars Shooting Percentages Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'FG%', color = 'Pos', title='All-Stars Shooting Percentages Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = '3P%', color = 'Pos', title='All-Stars Shooting Percentages Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'STL', color = 'Pos', title='All-Stars Steals Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'BLK', color = 'Pos', title='All-Stars Blocks Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'TRB', color = 'Pos', title='All-Stars Rebounds Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'AST', color = 'Pos', title='All-Stars Assists Distribution')

fig.show()
fig = px.scatter(df_combined, x = 'Player', y = 'PTS', color = 'Pos', title='All-Stars Points Distribution')

fig.show()