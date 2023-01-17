%matplotlib inline



import numpy as np

import pandas as pd

import sqlite3



import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import seaborn as sns
with sqlite3.connect('../input/database.sqlite') as con:

    matchs = pd.read_sql_query('select * from Match', con)

    countries = pd.read_sql_query('select * from Country', con)

    teams = pd.read_sql_query('select * from Team', con)
big5 = 'England', 'France', 'Germany', 'Italy', 'Spain'

bigLeagues = countries[countries.name.isin(big5)]

bigLeagues
desiredColumns = ['id', 'country_id', 'season', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A']

selectedMatches = matchs[matchs.league_id.isin(bigLeagues.id)]

selectedMatches = selectedMatches[desiredColumns]

selectedMatches.tail()
# detect corrupted data points

selectedMatches[(selectedMatches.B365H.isnull())].head()
# and drop 'em

selectedMatches = selectedMatches[np.isfinite(selectedMatches['B365H'])]
def oddsToProbs(row):

    odds = [row['B365H'], row['B365D'], row['B365A']]

    probs = [1. / o for o in odds]

    norm = sum(probs)

    probs = [p / norm for p in probs]

    return probs



def homePoints(row):

    pts = 1

    if row['home_team_goal'] > row['away_team_goal']:

        pts = 3

    elif row['home_team_goal'] < row['away_team_goal']:

        pts = 0

    return pts



def awayPoints(row):

    pts = 1

    if row['home_team_goal'] > row['away_team_goal']:

        pts = 0

    elif row['home_team_goal'] < row['away_team_goal']:

        pts = 3

    return pts



def homeTeamExp(row):

    probs = oddsToProbs(row)

    return (3 * probs[0] + probs[1])



def awayTeamExp(row):

    probs = oddsToProbs(row)

    return (3 * probs[2] + probs[1])



def homeTeamPerformance(row):

    return (homePoints(row) - homeTeamExp(row))



def awayTeamPerformance(row):

    return (awayPoints(row) - awayTeamExp(row))
selectedMatches['home_exp'] = selectedMatches.apply(homeTeamExp, axis=1)

selectedMatches['away_exp'] = selectedMatches.apply(awayTeamExp, axis=1)



selectedMatches['home_pfm'] = selectedMatches.apply(homeTeamPerformance, axis=1)

selectedMatches['away_pfm'] = selectedMatches.apply(awayTeamPerformance, axis=1)



selectedMatches.head()
exp = []

pfm = []

clr = []



# prepare for outliers

otSeason = []

otExp = []

otPfm = []

otName = []

hotSeason = []

hotExp = []

hotPfm = []

hotName = []

notSeason = []

notExp = []

notPfm = []

notName = []



colors = ['red', 'green', 'orange', 'purple', 'blue']

colors_mapping = dict(zip(bigLeagues.id, colors))



for season, seasonMatches in selectedMatches.groupby('season'):

    for teamID, teamName in zip(teams.team_api_id, teams.team_long_name):

        teamHM = seasonMatches[seasonMatches.home_team_api_id == teamID]

        teamAM = seasonMatches[seasonMatches.away_team_api_id == teamID]

        teamExp = teamHM['home_exp'].sum() + teamAM['away_exp'].sum()

        teamPfm = teamHM['home_pfm'].sum() + teamAM['away_pfm'].sum()

        if teamExp > 0:

            # rescale corrupted games, and Bundesliga (only 34 games per season)

            teamExp = teamExp / (teamHM['home_exp'].count() + teamAM['home_exp'].count()) * 38

            teamPfm = teamPfm / (teamHM['home_exp'].count() + teamAM['home_exp'].count()) * 38

            teamClr = colors_mapping[teamHM.country_id.values[0]]

            exp.append(teamExp)

            pfm.append(teamPfm)

            clr.append(teamClr)

            # keep outliers

            # have to do this stupid thing to avoid kaggle kernel issues

            if (teamPfm > 19.4) | (teamPfm < -16.4):

                otSeason.append(season)

                otExp.append(teamExp)

                otPfm.append(teamPfm)

                otName.append(teamName)

            if (teamPfm > 17.1) | ((teamExp > 78.8) & (teamPfm > 10)):

                hotSeason.append(season)

                hotExp.append(teamExp)

                hotPfm.append(teamPfm)

                hotName.append(teamName)

            if (teamPfm < -15.7) | ((teamExp > 60) & (teamPfm < -10)):

                notSeason.append(season)

                notExp.append(teamExp)

                notPfm.append(teamPfm)

                notName.append(teamName)

                

# simplify club names

nameDict = {'Hannover 96':'Hannover',

            'Montpellier Hérault SC':'Montpellier',

            'Real Betis Balompié':'Real Betis',

            'Borussia Dortmund':'Dortmund',

            'AJ Auxerre':'Auxerre',

            'Real Madrid CF':'Real Madrid',

            'FC Barcelona':'Barcelona',

            'Manchester City':'Man City',

            'Atlético Madrid':'Atlético',

            'Manchester United':'Man Utd',

            'FC Bayern Munich':'Bayern',

            'Hertha BSC Berlin':'Hertha',

            'Olympique de Marseille':'OM',

            'Queens Park Rangers':'QPR',

            'ES Troyes AC':'Troyes',

            'SV Werder Bremen':'Bremen'

            }

otName = [nameDict[n] if n in nameDict else n for n in otName]

hotName = [nameDict[n] if n in nameDict else n for n in hotName]

notName = [nameDict[n] if n in nameDict else n for n in notName]





# and season representations

import re

otSeason = [re.sub('(20)', '', s) for s in otSeason]

hotSeason = [re.sub('(20)', '', s) for s in hotSeason]

notSeason = [re.sub('(20)', '', s) for s in notSeason]
# plot

plt.figure(figsize=(16,12))

plt.scatter(exp, pfm, color=clr, s=[60]*len(exp))

plt.title('2008-2016: True Performance against Expectation', fontsize=16)



# legend

circles = []

labels = []

for leagueID, name in zip(bigLeagues.id, bigLeagues.name):

    labels.append(name)

    circles.append(Line2D([0], [0], linestyle="none", marker="o", markersize=8, markerfacecolor=colors_mapping[leagueID]))

plt.legend(circles, labels, numpoints=1, loc=(0., 0.85), prop={'size': 12})



plt.ylim((-30,30))

plt.xlim((20,100))



# annotation

plt.xlabel('Expected Points', fontsize=14)

plt.ylabel('True Performance', fontsize=14)

ax = plt.gca()



# offset

xoffset = 0.4

yoffset = -0.3



#annotate outliers

for ind, Pfm in enumerate(otPfm):

    ax.annotate(otSeason[ind]+' '+otName[ind], xy=(otExp[ind]+xoffset, Pfm+yoffset), fontsize=14)
# plot

plt.figure(figsize=(14, 8))

plt.scatter(exp, pfm, color=clr, s=[60]*len(exp))

plt.title('2008-2016: Outperformers', fontsize=16)



# legend

circles = []

labels = []

for leagueID, name in zip(bigLeagues.id, bigLeagues.name):

    labels.append(name)

    circles.append(Line2D([0], [0], linestyle="none", marker="o", markersize=8, markerfacecolor=colors_mapping[leagueID]))

plt.legend(circles, labels, numpoints=1, loc=(0., 0.78), prop={'size': 12})



plt.ylim((10,30))

plt.xlim((30,100))



# annotation

plt.xlabel('Expected Points', fontsize=14)

plt.ylabel('True Performance', fontsize=14)

ax = plt.gca()



# offset

xoffset = 0.4

yoffset = -0.15



#annotate outliers

for ind, Pfm in enumerate(hotPfm):

    ax.annotate(hotSeason[ind]+' '+hotName[ind], xy=(hotExp[ind]+xoffset, Pfm+yoffset), fontsize=14)





ax.annotate('12/13 Man Utd', xy=(69, 11.2), fontsize=14)

ax.annotate('13/14 Atlético', xy=(70, 11.75), fontsize=14)

ax.annotate('11/12 Dortmund', xy=(76.9, 13.9), fontsize=14)



ax.annotate('08/09 Wolfsburg', xy=(62.3, 15.2), fontsize=14)

ax.annotate('08/09 Genoa', xy=(41.5, 17), fontsize=14)

plt.plot([49.2, 51.7], [17, 16], 'k-', lw=0.75)

ax.annotate('08/09 Hertha', xy=(55, 16), fontsize=14)

ax.annotate('10/11 Mainz, 11/12 Newcastle', xy=(32, 16), fontsize=14)   

# plot

plt.figure(figsize=(14, 8))

plt.scatter(exp, pfm, color=clr, s=[60]*len(exp))

plt.title('2008-2016: Les Misérables', fontsize=16)



# legend

circles = []

labels = []

for leagueID, name in zip(bigLeagues.id, bigLeagues.name):

    labels.append(name)

    circles.append(Line2D([0], [0], linestyle="none", marker="o", markersize=8, markerfacecolor=colors_mapping[leagueID]))

plt.legend(circles, labels, numpoints=1, loc=(0., 0.), prop={'size': 12})



plt.ylim((-25,-5))

plt.xlim((25,95))



# annotation

plt.xlabel('Expected Points', fontsize=14)

plt.ylabel('True Performance', fontsize=14)

ax = plt.gca()



# offset

xoffset = 0.4

yoffset = -0.2



#annotate outliers

for ind, Pfm in enumerate(notPfm):

    if ((notName[ind] != 'Troyes') & (notName[ind] != 'QPR') & (notName[ind] != 'Bremen')):

        ax.annotate(notSeason[ind]+' '+notName[ind], xy=(notExp[ind]+xoffset, Pfm+yoffset), fontsize=14)



ax.annotate('15/16 Troyes', xy=(26.2, -16.3), fontsize=14)

ax.annotate('12/13     \nQPR', xy=(37.2, -16.6), fontsize=14)

ax.annotate('08/09 Bremen', xy=(64.4, -13.8), fontsize=14)