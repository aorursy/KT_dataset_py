import json

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)


raw_data_path = '''../input/champion_round_base_analysis.json'''



with open(raw_data_path) as f:

    data = json.load(f)



df = pd.DataFrame(data).sort_index(axis=1).sort_index()



df.head()
# Number of comps that include a given unit, by round

df_count = df.applymap(lambda x: x['count'] if isinstance(x, dict) else 0)



df_count.head()
# Average placement of players that used a given unit, by round

df_avg_place = df.applymap(lambda x: x['avg_place'] if isinstance(x, dict) else np.nan)



df_avg_place.head()
df_avg_place_noRound = df_avg_place['total'].drop('total').sort_values(ascending=False)

ax = df_avg_place_noRound.plot.barh(figsize=(5,20), fontsize=15)

ax.figure.tight_layout()

ax.set_title('Champion Average Placement', fontdict={'fontsize': 20 } )

ax
# Transforms a round (33) into a stage (4-5)

def round_to_stage(round):

    return f'{int((round + 9) / 7)}-{((round + 2) % 7 + 1)}'
df_count_LucianAndThresh = df_count.loc[['Lucian', 'Thresh']].drop('total',axis=1).T

df_count_LucianAndThresh.index = pd.Series(df_count_LucianAndThresh.index).apply(lambda x: round_to_stage(int(x)))

ax = df_count_LucianAndThresh.plot(figsize=(12,6))

ax.figure.tight_layout()

ax.set_title('Champion frequency', fontdict={'fontsize': 20 } )

ax.set_xlabel('Round the player finished the game', fontdict={'fontsize': 15 } )

ax.set_ylabel('Number of players using the champion', fontdict={'fontsize': 15 } )

ax
#Champion placement relative to the average, by round

df_avg_place_relative = df_avg_place.copy()



champions = [x for x in df_avg_place_relative.index if not x == 'total']



for champion in champions:

    df_avg_place_relative.loc[champion] = df_avg_place_relative.loc['total'] - df_avg_place_relative.loc[champion]

    

df_avg_place_relative.head()
df_avg_place_relative_LucianAndThresh = df_avg_place_relative.loc[['Lucian', 'Thresh']]

df_avg_place_relative_LucianAndThresh = df_avg_place_relative_LucianAndThresh[[str(x) for x in range(23,45) if str(x) in df_avg_place_relative_LucianAndThresh.columns]].T

df_avg_place_relative_LucianAndThresh.index = pd.Series(df_avg_place_relative_LucianAndThresh.index).apply(lambda x: round_to_stage(int(x)))

df_avg_place_relative_LucianAndThresh['Round average'] = df_avg_place_relative_LucianAndThresh['Lucian'].apply(lambda x: 0)

df_avg_place_relative_LucianAndThresh

ax = df_avg_place_relative_LucianAndThresh.plot(figsize=(12,6))

ax.figure.tight_layout()

ax.set_title('Champion placement (relative to round average)', fontdict={'fontsize': 20 } )

ax.set_ylabel('Relative placement ((champion - total) / total)', fontdict={'fontsize': 15 } )

ax.set_xlabel('Round', fontdict={'fontsize': 15 } )

ax
#Champion placement relative to the average and weighted by their frequency, by round

df_avg_place_weighted_relative = df_avg_place_relative.copy()



champions = [x for x in df_avg_place_relative.index if not x == 'total']



for champion in champions:

    df_avg_place_weighted_relative.loc[champion] = df_avg_place_weighted_relative.loc[champion] * df_count.loc[champion]

    df_avg_place_weighted_relative.loc[champion] = df_avg_place_weighted_relative.loc[champion] / df_count['total'][champion]



df_avg_place_weighted_relative.head()
df_total_players = df_count.loc['total'].drop('total')

df_total_players.index = pd.Series(df_total_players.index).apply(lambda x: round_to_stage(int(x)))



ax = df_total_players.plot(figsize=(12,6))

ax.set_title('Players recorded in our dataset', fontdict={'fontsize': 20 } )

ax.set_ylabel('Number of players', fontdict={'fontsize': 15 } )

ax.set_xlabel('Round', fontdict={'fontsize': 15 } )

ax
df_avg_place_relative_LucianAndThresh = df_avg_place_weighted_relative.loc[['Lucian', 'Thresh']]

df_avg_place_relative_LucianAndThresh = df_avg_place_relative_LucianAndThresh[[str(x) for x in range(23,45) if str(x) in df_avg_place_relative_LucianAndThresh.columns]].T

df_avg_place_relative_LucianAndThresh.index = pd.Series(df_avg_place_relative_LucianAndThresh.index).apply(lambda x: round_to_stage(int(x)))

df_avg_place_relative_LucianAndThresh['Round average'] = df_avg_place_relative_LucianAndThresh['Lucian'].apply(lambda x: 0)

df_avg_place_relative_LucianAndThresh

ax = df_avg_place_relative_LucianAndThresh.plot(figsize=(12,6))

ax.figure.tight_layout()

ax.set_title('Champion placement (relative and weighted)', fontdict={'fontsize': 20 } )

ax.set_ylabel('Relative placement ((champion - total) / total)', fontdict={'fontsize': 15 } )

ax.set_xlabel('Round', fontdict={'fontsize': 15 } )

ax
df_avg_place_relative_LucianAndThresh = df_avg_place_weighted_relative.loc[['Lucian', 'Darius']]

df_avg_place_relative_LucianAndThresh = df_avg_place_relative_LucianAndThresh[[str(x) for x in range(23,45) if str(x) in df_avg_place_relative_LucianAndThresh.columns]].T

df_avg_place_relative_LucianAndThresh.index = pd.Series(df_avg_place_relative_LucianAndThresh.index).apply(lambda x: round_to_stage(int(x)))

df_avg_place_relative_LucianAndThresh['Round average'] = df_avg_place_relative_LucianAndThresh['Lucian'].apply(lambda x: 0)

df_avg_place_relative_LucianAndThresh

ax = df_avg_place_relative_LucianAndThresh.plot(figsize=(12,6))

ax.figure.tight_layout()

ax.set_title('Champion placement (relative and weighted)', fontdict={'fontsize': 20 } )

ax.set_ylabel('Relative placement ((champion - total) / total)', fontdict={'fontsize': 15 } )

ax.set_xlabel('Round', fontdict={'fontsize': 15 } )

ax