%matplotlib inline



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



df_hitters = pd.read_csv("../input/mlb_2017_regular_season_top_hitting.csv")



avg = df_hitters["AVG?"]

obp = df_hitters['OBP']

slg = df_hitters["SLG"]

obpps = df_hitters['OPS']



f, axes = plt.subplots(2, 2, figsize=(15, 10),sharey=True)

    

sns.distplot(avg, bins=20, kde=False, color = "b", ax=axes[0, 0]).set_title("2017 Top 144 Hitters: Batting Average")

sns.distplot(obp, bins=20, kde=False, color = "r", ax=axes[0, 1]).set_title("2017 Top 144 Hitters: On Base Percentage")

sns.distplot(slg, bins=20, kde=False, color = "g", ax=axes[1, 0]).set_title("2017 Top 144 Hitters: Slugging Percentage")

sns.distplot(obpps, bins=20, kde=False, color = "m", ax=axes[1, 1]).set_title("2017 Top 144 Hitters: On Base Plus Slugging")

f, axes[0,0].set_xlabel('Avg')

f, axes[0,0].set_ylabel('# Hitters')

f, axes[0,1].set_xlabel('OBP')

f, axes[1,0].set_xlabel('SLG')

f, axes[1,0].set_ylabel('# Hitters')

f, axes[1,1].set_xlabel('OPS')