import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sqlite3

import numpy as np

from numpy import random



#load data (make sure you have downloaded database.sqlite)

with sqlite3.connect('../input/database.sqlite') as con:

    matches = pd.read_sql_query("SELECT * from Match", con)

    
matches = matches[['home_team_goal', 'away_team_goal','B365H', 'B365D' ,'B365A']]

matches.dropna(inplace=True)

matches.head()
def f(row):

    if row['home_team_goal'] == row['away_team_goal']:

        val = 0

    elif row['home_team_goal'] > row['away_team_goal']:

        val = 1

    else:

        val = 2

    return val



matches['winner'] = matches.apply(f, axis=1)



matches.head()
matches['winner'].value_counts(normalize=True)
print(matches.head())