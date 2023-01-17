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

print(barcelona.team_api_id)
cur.execute("select home_team_goal, away_team_goal from Match where league_id = 21518 AND home_team_api_id = 8634 AND (season = '2008/2009')").fetchall()
cur.execute("select home_team_goal, away_team_goal from Match where league_id = 21518 AND away_team_api_id = 8634 AND (season = '2008/2009')").fetchall()
cur.execute("select DISTINCT season from Match").fetchall()
x = np.arange(10)

df = pd.DataFrame(np.random.randint(0, 10, size=(10, 3)))

print(len(x))

print(len(df[0]))

sns.barplot(x, df[0])
sns.factorplot(x, df)