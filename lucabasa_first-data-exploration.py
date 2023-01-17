# standard
import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option("display.max_columns",None)
df = pd.read_csv('../input/bouts_out_new.csv')
df.head()
df.info()
df.describe()
fil = ((df.height_A < 224) & (df.height_A > 147) &
      (df.height_B < 224) & (df.height_B > 147) &
      (df.weight_B > 70) & (df.weight_A > 70) &
      (df.age_A < 60) & (df.age_A > 14) &
      (df.age_B < 60) & (df.age_B > 14) &
      (df.reach_A < 250) & (df.reach_A > 130) &
      (df.reach_B < 250) & (df.reach_B > 130)) 
df = df[fil]
df.info()
fig, ax = plt.subplots(4,2, figsize=(12, 20))
sns.distplot(df.age_A, bins = 20, ax=ax[0][0])
sns.distplot(df.age_B, bins = 20, ax=ax[0][1])
sns.distplot(df.height_A, bins = 50, ax=ax[1][0])
sns.distplot(df.height_B, bins = 50, ax=ax[1][1])
sns.distplot(df.weight_A, bins = 50, ax=ax[2][0])
sns.distplot(df.weight_B, bins = 50, ax=ax[2][1])
sns.distplot(df.reach_A, bins = 50, ax=ax[3][0])
sns.distplot(df.reach_B, bins = 50, ax=ax[3][1])
df['Diff_age'] = df.age_A - df.age_B
df[['Diff_age', 'result']].groupby('result').mean()
g = sns.FacetGrid(df, hue='result', size = 7)
g.map(plt.scatter, 'age_A', 'age_B', edgecolor="w")
g.add_legend()
df['Diff_weight'] = df.weight_A - df.weight_B
df[['Diff_weight', 'result']].groupby('result').mean()
g = sns.FacetGrid(df, hue='result', size = 7)
g.map(plt.scatter, 'weight_A', 'weight_B', edgecolor="w")
g.add_legend()
df['Diff_height'] = df.height_A - df.height_B
df[['Diff_height', 'result']].groupby('result').mean()
g = sns.FacetGrid(df, hue='result', size = 7)
g.map(plt.scatter, 'height_A', 'height_B', edgecolor="w")
g.add_legend()
df['Diff_reach'] = df.reach_A - df.reach_B
df[['Diff_reach', 'result']].groupby('result').mean()
g = sns.FacetGrid(df, hue='result', size = 7)
g.map(plt.scatter, 'reach_A', 'reach_B', edgecolor="w")
g.add_legend()
df['Tot_fight_A'] = df.won_A + df.lost_A + df.drawn_A
df['Tot_fight_B'] = df.won_B + df.lost_B + df.drawn_B
df['Diff_exp'] = df.Tot_fight_A - df.Tot_fight_B
df[['Diff_exp', 'result']].groupby('result').mean()
df.Diff_exp.hist(bins= 100)
g = sns.FacetGrid(df, hue='result', size = 7)
g.map(plt.scatter, 'Tot_fight_A', 'Tot_fight_B', edgecolor="w")
g.add_legend()
df['Win_per_A'] = df.won_A / df.Tot_fight_A
df.loc[df.Tot_fight_A == 0, 'Win_per_A'] = 0 #because maybe it is the first fight
df['Win_per_B'] = df.won_B / df.Tot_fight_B
df.loc[df.Tot_fight_B == 0, 'Win_per_B'] = 0
df['KO_perc_A'] = df.kos_A / df.won_A
df.loc[df.won_A == 0, 'KO_perc_A'] = 0
df['KO_perc_B'] = df.kos_B / df.won_B
df.loc[df.won_B == 0, 'KO_perc_B'] = 0
fig, ax = plt.subplots(1,2, figsize=(12, 5))
sns.distplot(df.Win_per_A, bins = 50, ax=ax[0])
sns.distplot(df.Win_per_B, bins = 50, ax=ax[1])
df.KO_perc_B.hist(bins = 30)
df.KO_perc_A.hist(bins = 30)
df.loc[df.stance_A == df.stance_B, 'Stance'] = 0
df.loc[(df.stance_A == 'orthodox') & (df.stance_B == 'southpaw'), 'Stance'] = 1
df.loc[(df.stance_B == 'orthodox') & (df.stance_A == 'southpaw'), 'Stance'] = -1
pd.crosstab(df.Stance, df.result)
df[df.stance_A != df.stance_B].stance_B.value_counts(dropna = False)