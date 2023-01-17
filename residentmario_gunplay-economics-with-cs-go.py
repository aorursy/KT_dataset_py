import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
plt_kwargs = {'figsize': (10, 4)}
weapon_events = pd.read_csv("../input/mm_master_demos.csv", index_col=0)
weapon_events['round_type'].value_counts()
fig = plt.figure(figsize=(10, 5))
sns.kdeplot(weapon_events['seconds'])
plt.suptitle("Weapon Event Time, Seconds into the Match")
weapon_events['round_type'].value_counts().plot.bar(title='Round Types', **plt_kwargs)
fig = plt.figure(figsize=(10, 5))
sns.kdeplot(weapon_events['ct_eq_val'].rename('Counter-Terrorists'))
sns.kdeplot(weapon_events['t_eq_val'].rename('Terrorists'))
plt.suptitle("Team Round Spend Values")
match_level_data = weapon_events.groupby('file').head()
df = pd.DataFrame().assign(winner=match_level_data['winner_side'], point_diff=match_level_data['ct_eq_val'] - match_level_data['t_eq_val'])
df = df.assign(point_diff=df.apply(lambda srs: srs.point_diff if srs.winner[0] == 'C' else -srs.point_diff, axis='columns'), winner=df.winner.map(lambda v: True if v[0] == 'C' else False))

df = (df
     .assign(point_diff_cat=pd.qcut(df.point_diff, 10))
     .groupby('point_diff_cat')
     .apply(lambda df: df.winner.sum() / len(df.winner))
)
df.index = df.index.values.map(lambda inv: inv.left + (inv.right - inv.left) / 2).astype(int)

fig = plt.figure(figsize=(10, 5))
df.plot.line()
plt.suptitle("Play Advantage Created by Additional Spend")
ax = plt.gca()
ax.axhline(0.5, color='black')
ax.set_ylim([0, 1])
ax.set_xlabel('Spend')
ax.set_ylabel('% Games Won')
fig = plt.figure(figsize=(10, 5))

sns.kdeplot(match_level_data.query('winner_side == "CounterTerrorist"').pipe(lambda df: df.ct_eq_val - df.t_eq_val).rename('Winning Matches'))
sns.kdeplot(match_level_data.pipe(lambda df: df.ct_eq_val - df.t_eq_val).rename('All Matches'))

plt.suptitle("Team Weapon Values")
g = sns.FacetGrid(weapon_events.assign(
    total_val=weapon_events['ct_eq_val'] + weapon_events['t_eq_val']
), col="wp_type", col_wrap=4)
g.map(sns.kdeplot, 'total_val')