import numpy as np 

import pandas as pd 

import os



# read file

df = pd.read_csv('../input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv')

df = df[['League', 'Season']].drop_duplicates(ignore_index=True)

df['active'] = 1

df['Season Start'] = df['Season'].str[:4]
import seaborn as sns

import matplotlib.pyplot as plt



heatmap1_data = pd.pivot_table(df, values='active', 

                     index=['League'], 

                     columns='Season Start')





plt.figure(figsize=(12, 24)) 

plt.title("Seasons per League", fontsize =20)





sns.heatmap(heatmap1_data, cmap="RdYlGn",linewidths = .5,cbar=False,xticklabels=1)
