import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

sns.set(style = 'white')
conditions = pd.read_csv("../input/triple-crown-of-horse-races-2005-2019/TrackConditions.csv")

races = pd.read_csv("../input/triple-crown-of-horse-races-2005-2019/TripleCrownRaces_2005-2019.csv")
races.head(20)
races.dtypes
conditions.head(20)
conditions.dtypes
df = races.merge(conditions, on = ['year', 'race'])
df.head()
plt.hist(df['Odds'])

plt.xlabel('Odds')
df1 = df[df['final_place'] == 1]

plt.scatter(df1['Odds'], df1['Win'])
df['final_place_cat'] = pd.cut(df['final_place'], [0, 3, 22], labels = ['Top 3', 'Rest'], right = True) 



grouped_df = df.groupby(['final_place_cat', 'track_condition'])['Odds'].mean().reset_index()



barwidth = 0.25



bars1 = grouped_df['Odds'][grouped_df['track_condition'] == 'Fast']

bars2 = grouped_df['Odds'][grouped_df['track_condition'] == 'Muddy']

bars3 = grouped_df['Odds'][grouped_df['track_condition'] == 'Sloppy']



r1 = np.arange(len(bars1))

r2 = [x + barwidth for x in r1]

r3 = [x + barwidth for x in r2]



fig, ax = plt.subplots(figsize = (10,6))

fig.tight_layout()

fig.subplots_adjust(bottom = 0.25, top = 0.9)

ax.bar(r1, bars1, color = 'b', width = barwidth, label = 'Fast')

ax.bar(r2, bars2, color = 'r', width = barwidth, label = 'Muddy')

ax.bar(r3, bars3, color = 'g', width = barwidth, label = 'Sloppy')

ax.set_ylabel('Starting Odds', fontsize = 16)

ax.set_yticklabels([0, 5, 10, 15, 20, 25], fontsize = 16)

ax.set_xlabel('Finishing Place', fontsize = 16)

ax.set_xticks([0.25, 1.25])

ax.set_xticklabels(['Top 3', 'Rest of Field'], fontsize = 16)

fig.legend(loc = 'lower center', ncol = 3, facecolor = 'white', edgecolor = 'white', fontsize = 16)

fig.suptitle("Starting Odds of Top 3 vs. Rest of the Field Based on Track Conditions", fontsize = 16)
