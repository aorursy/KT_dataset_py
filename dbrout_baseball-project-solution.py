import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
pitches = pd.read_csv('../input/pitches.csv')

pitches.rename(columns={'ab_id':'atbat_id'}, inplace=True)

print(pitches.shape)

print(pitches.columns)

pitches.head(2)
atbats = pd.read_csv('../input/atbats.csv')

atbats.rename(columns={'ab_id':'atbat_id'}, inplace=True)

print(atbats.shape)

atbats.head(2)
player_name = pd.read_csv('../input/player_names.csv')

player_name.rename(columns={'id':'batter_id'}, inplace=True)

print(player_name.shape)

player_name.head()
player_name['Full_Name'] = player_name['first_name']+' '+player_name['last_name']



pitches['speed_diff'] = pitches['start_speed']-pitches['end_speed']



#the pitch result (ball or strike) is not provided so we have to compute it

pitches['isball'] = (pitches['b_count']- pitches['b_count'].shift(-1)) < 0



pitches.head(30)

pitches = pitches[pitches['isball']!=True]

pitches.head()
atbats['hit'] = atbats['event'].map(lambda row: not 'out' in row).astype(int)

atbats.head()
merged1_df = pd.merge(pitches, atbats,  how='left', left_on='atbat_id', right_on = 'atbat_id')
big_df = pd.merge(merged1_df, player_name,  how='left', left_on='batter_id', right_on = 'batter_id')

big_df.head()
groupedby_pid = big_df.groupby(['pitcher_id']).mean()

count = big_df.groupby(['pitcher_id']).count()

groupedby_pid = groupedby_pid[count['hit']>100]
groupedby_pid.head()
plt.figure(figsize=(10,6))

ax = sbn.pairplot(groupedby_pid[['start_speed','speed_diff','spin_rate','break_length','hit']],kind="reg",plot_kws={'scatter_kws':{'s':2}});
atbat_results = big_df[big_df['end_speed']>90.].groupby(['batter_id','atbat_id']).aggregate('mean')

atbat_results.head(20)
best_batters = atbat_results[atbat_results.groupby(['batter_id']).count()>40].dropna().groupby(['batter_id'])[['hit']].mean()

best_batters.head(10)
best_batters_withnames = pd.merge(best_batters, player_name,  how='left', left_on='batter_id', right_on = 'batter_id')
best_batters_withnames.sort_values(by='hit',ascending=False)[:20].plot.bar(x='Full_Name',y='hit',figsize=(15,8), ylim=(.4,.6),rot=45);