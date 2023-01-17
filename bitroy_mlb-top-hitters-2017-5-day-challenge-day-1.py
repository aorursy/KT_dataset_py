%matplotlib inline  



import pandas as pd

import matplotlib.pyplot as plt
df_hitters = pd.read_csv("../input/mlb_2017_regular_season_top_hitting.csv")
df2 = df_hitters.columns

df2
team_counts = df_hitters['Team'].value_counts()

team_counts
font_config = {

    'font.size': 16,

}

with plt.style.context(font_config):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax = (df_hitters.groupby('Team')

      .HR.sum()

      .plot.bar())

    ax.set_ylabel('Home Runs')

    ax.set_title('2017 Home Runs by Team for Top 144 Hitters')
with plt.style.context(font_config):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax = (df_hitters.groupby('Team')

      .SB.sum()

      .plot.bar())

    ax.set_ylabel('Stolen Bases')

    ax.set_title('2017 Stolen Bases by Team for Top 144 Hitters')
hitstats = df_hitters.describe()

hitstats