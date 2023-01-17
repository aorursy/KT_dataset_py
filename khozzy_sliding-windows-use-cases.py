import numpy as np
import pandas as pd
def sliding(df, window, col, func):
    results = np.array([])
    
    for index, row in df.iterrows():
        snap = df.loc[index-window:index-1, :]
        results = np.append(results, [func(snap[col], )])
        
    ret = pd.Series(data=results, index=df.index)
    return ret
data = pd.read_csv('../input/data.csv')
data.sort_values(['game_id', 'game_event_id'], inplace=True)
last_shot_made = data.groupby('game_id').apply(lambda x: sliding(x, window=1, col='shot_made_flag', func=np.mean))
last_shot_made.index = last_shot_made.index.get_level_values(1) # Because we get back a multindex
last_shot_made[np.isnan(last_shot_made)] = 0.5 # Assume that first shots and posterior to unknowns has 0.5 probability
last_3_mean = data.groupby('game_id').apply(lambda x: sliding(x, window=3, col='shot_made_flag', func=np.mean))
last_3_mean.index = last_3_mean.index.get_level_values(1)
last_3_mean[np.isnan(last_3_mean)] = last_3_mean.mean() # replace NAN-s with overall mean
last_5_sum = data.groupby('game_id').apply(lambda x: sliding(x, window=5, col='shot_made_flag', func=np.sum))
last_5_sum.index = last_5_sum.index.get_level_values(1)
streak_5 = last_5_sum == 5.0
data['last_shot_made'] = last_shot_made
data['last_3_mean'] = last_3_mean
data['5_shots_streak'] = streak_5

data.loc[:, ['game_id', 'game_event_id', 'shot_made_flag', 'last_shot_made', 'last_3_mean', '5_shots_streak']].head(30)