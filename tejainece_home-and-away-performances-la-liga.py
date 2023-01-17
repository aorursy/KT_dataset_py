import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns

import typing
conn = sqlite3.connect('../input/database.sqlite')

cur = conn.cursor()
def filterTeam(teamName):

    return cur.execute("select * from Team where team_long_name like '%{}%'".format(teamName)).fetchall()



print(filterTeam('Barcelona'))
class Team:

    def __init__(self, tup: typing.Tuple):

        self.team_api_id = tup[1]

        self.team_long_name = tup[3]

        self.team_short_name = tup[4]

        

    @classmethod

    def byTeamName(self, teamName):

        team = filterTeam(teamName)

        if len(team) == 0:

            return None

        return Team(team[0])

        

    def __str__(self):

        return self.team_long_name



barcelona = Team.byTeamName('Barcelona')

print(barcelona)



realMadrid = Team.byTeamName('Real Madrid')

print(realMadrid)



atleticoMadrid = Team.byTeamName('Atlético Madrid')

print(atleticoMadrid)



sevilla = Team.byTeamName('Sevilla')

print(sevilla)



villarreal = Team.byTeamName('Villarreal')

print(villarreal)



sevilla = Team.byTeamName('Sevilla')

print(sevilla)
class Performance:

    homeGoalsByTeamQry = 'select home_team_goal from Match where home_team_api_id = {}'

    awayGoalsByTeamQry = 'select away_team_goal from Match where away_team_api_id = {}'



    def __init__(self, team: Team):

        self.goals_home = np.array(cur.execute(Performance

                                               .homeGoalsByTeamQry

                                               .format(team.team_api_id))

                                   .fetchall())[:, 0]

        self.goals_away = np.array(cur.execute(Performance

                                               .awayGoalsByTeamQry

                                               .format(team.team_api_id))

                                   .fetchall())[:, 0]

        self.diff = self.goals_home - self.goals_away

        self.df = pd.DataFrame({'home': self.goals_home.tolist(), 

                                'away': self.goals_away.tolist(), 

                                'diff': self.diff.tolist()})
print('Barcelona stats')

barcelonaPref = Performance(barcelona)

print(barcelonaPref.df.sum(axis=0))

print()



print('Real Madrid stats')

realMadridPref = Performance(realMadrid)

print(realMadridPref.df.sum(axis=0))

print()



print('Athletico Madrid stats')

realMadridPref = Performance(realMadrid)

print(realMadridPref.df.sum(axis=0))

print()



print('Villarreal stats')

villarrealPref = Performance(villarreal)

print(villarrealPref.df.sum(axis=0))

print()



print('Sevilla stats')

sevillaPref = Performance(sevilla)

print(sevillaPref.df.sum(axis=0))

print()