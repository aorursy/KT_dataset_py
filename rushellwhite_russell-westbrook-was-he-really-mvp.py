import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode,iplot,iplot_mpl,download_plotlyjs,enable_mpl_offline

import plotly.graph_objs as go

import os



%matplotlib inline



init_notebook_mode(connected=True)
players = pd.read_csv("../input/Players.csv",usecols=["Player","height","weight","collage","born","birth_city","birth_state"])

players[["height","weight","born"]] = players[["height","weight","born"]].fillna(0).astype(int)

season = pd.read_csv("../input/Seasons_Stats.csv")

season[["Year","Age","G","GS","AST","STL","BLK","PTS","PF","TOV","FT","FTA","2P","2PA","3P","3PA","ORB","TRB","DRB"]] = season[["Year","Age","G","GS","AST","STL","BLK","PTS","PF","TOV","FT","FTA","2P","2PA","3P","3PA","ORB","TRB","DRB"]].fillna(0).astype(int)

season = season[season["Year"] >= 2010]

season["PPG"] = season["PTS"] / season["G"]

season["APG"] = season["AST"] / season["G"]

season["3PG"] = season["3P"] / season["G"]

season["RPG"] = season["TRB"] / season["G"]

season["FTPG"] = season["FTA"] / season["G"]

season["TOPG"] = season["TOV"] / season["G"]

season["PPG"] = round(season["PPG"],1)
pd.options.mode.chained_assignment = None  # default='warn'
stats = season[["Year","Player","Pos","Age","Tm","PER","PTS","PPG","APG","3PG","RPG","FTPG","TOPG","FG%","3P%","2P%","FT%",'WS','BPM','OWS','DWS']].reset_index()
seventeen = stats[(stats['Year'] == 2017) & (stats['PPG'] >= 15)]
def changePosition(x):

    if ("SG" in x) or ("PG" in x):

        return "Guard"

    elif ("SF" in x) or ("PF" in x):

        return "Forward"

    else:

        return "Center"
seventeen['Position'] = seventeen['Pos'].apply(changePosition)
mask1 = (seventeen['Player'] == 'Lou Williams') & (seventeen['Tm'] != 'TOT')

mask2 = (seventeen['Player'] == 'DeMarcus Cousins') & (seventeen['Tm'] != 'TOT')

seventeen = seventeen[~(mask1 | mask2)]
# fig = plt.figure(figsize=(12,8))

# ax = fig.add_axes([0,0,.7,.7])

# ax.set_title('WS vs. Pts')

trace = go.Scatter(x = seventeen['WS'],y = seventeen['PPG'],mode = 'markers+text',text=seventeen['Player'],textposition = 'right')



data = [trace]

layout = dict(title = 'WS vs. PPG',

              hovermode = 'closest',

              yaxis = dict(zeroline = False,title="PPG"),

              xaxis = dict(zeroline = False,range=(0,18),title="Win Share")

             )



fix = dict(data=data,layout=layout)



iplot(fix,filename='basic-scatter')
fig = plt.figure(figsize=(12,8))

axes = fig.add_axes([0,0,1,1])

sns.barplot(x="BPM",y="Player",data=seventeen.sort_values("BPM",ascending=False).head(20))
guards = seventeen[seventeen['Position'] == 'Guard'].sort_values('PER',ascending=False).head(25)

forwards = seventeen[seventeen['Position'] == 'Forward'].sort_values('PER',ascending=False).head(25)

centers = seventeen[seventeen['Position'] == 'Center'].sort_values('PER',ascending=False).head(25)

guard = go.Box(y=guards['PER'],name="Guards",boxpoints='outliers')

forward = go.Box(y=forwards['PER'],name="Forwards")

center = go.Box(y=centers['PER'],name="Centers")



data = [guard,forward,center]

iplot(data)
fig = plt.figure(figsize=(12,8))

fig.add_subplot(1,2,1)

sns.barplot(x="PER",y="Player",data=guards.sort_values('PER',ascending=False))



fig.add_subplot(1,2,2)

sns.barplot(x="PER",y="Player",data=forwards.sort_values('PER',ascending=False),orient='h')



plt.tight_layout()

plt.show()
def hollingerMVP(x):

    if x >= 35.0:

        return '1. All-Time Great Season'

    elif x >= 30.0 and x < 35:

        return '2. Runaway MVP Candidate'

    elif x >= 27.5 and x < 30:

        return '3. Strong MVP Candidate'

    elif x >= 25.0 and x < 27.5:

        return '4. Weak MVP Candidate'

    else:

        return "Not MVP Candidate"
seventeen['MVP Candidacy'] = seventeen['PER'].apply(hollingerMVP)
mvpCandidates = seventeen[seventeen['MVP Candidacy'] != 'Not MVP Candidate']
mvpCandidates = mvpCandidates.sort_values("MVP Candidacy").reset_index()

mvpCandidates.drop(["level_0","index"],axis=1,inplace=True)
mvpCandidates.head(8)
sixteen = []

fifteen = []

fourteen = []

thirteen = []

twelve = []

eleven = []

current = []

for i,row in season.sort_values('PER',ascending=False).iterrows():

    if row['Year'] == 2016 and row['MP'] > 2000:

        sixteen.append(row)

    elif row['Year'] == 2015 and row['MP'] > 2000:

        fifteen.append(row)

    elif row['Year'] == 2014 and row['MP'] > 2000:

        fourteen.append(row)

    elif row['Year'] == 2013 and row['MP'] > 2000:

        thirteen.append(row)

    elif row['Year'] == 2017 and row['MP'] > 2000:

        current.append(row)

    elif row['Year'] == 2012 and row['MP'] > 2000:

        twelve.append(row)

    elif row['Year'] == 2011 and row['MP'] > 2000:

        eleven.append(row)

sixteen = pd.DataFrame(sixteen,columns=season.columns).head(5)

fifteen = pd.DataFrame(fifteen,columns=season.columns).head(5)

fourteen = pd.DataFrame(fourteen,columns=season.columns).head(5)

thirteen = pd.DataFrame(thirteen,columns=season.columns).head(5)

current = pd.DataFrame(current,columns=season.columns).head(5)

eleven = pd.DataFrame(eleven,columns=season.columns).head(5)

twelve = pd.DataFrame(twelve,columns=season.columns).head(5)

masterFrame = pd.concat([current,sixteen,fourteen,thirteen,fifteen,eleven,twelve])
masterFrame['MVP Candidacy'] = masterFrame['PER'].apply(hollingerMVP)
masterFrame = masterFrame.sort_values(["Year","MVP Candidacy"],ascending=True)

df = masterFrame[['Year','Player','PER','PPG','APG','RPG','MVP Candidacy','WS']].reset_index()

df
def mvp(player,year):

    if ('Curry' in player) & (year == 2016):

        return 'MVP'

    elif ('Curry' in player) and (year == 2015):

        return 'MVP'

    elif ('Westbrook' in player) and (year == 2017):

        return 'MVP'

    elif ('Durant' in player) and (year == 2014):

        return 'MVP'

    elif ('LeBron' in player) and (year == 2013):

        return 'MVP'

    elif ('LeBron' in player) and (year == 2012):

        return 'MVP'

    elif ('Rose' in player) and (year == 2011):

        return 'MVP'
df['MVP?'] = df.apply(lambda x:mvp(x['Player'],x['Year']),axis=1)
def highlight(val):

    color = 'red' if val == 'MVP' else 'black'

    return 'color: %s' % color
export = df.style.applymap(highlight)
df
trace = go.Scatter(x=df['Year'],y=df['PER'],mode = 'markers+text',text=df['Player'],textposition = 'right')

layout = dict(title = "MVP",hovermode='closest')

data = [trace]

fig = go.Figure(data=data,layout=layout)

iplot(fig)