import pandas as pd
df = pd.read_csv('../input/data.csv')
df.columns
from datetime import datetime
x =  df['game_date'].str.replace('-','')
from datetime import timedelta

df['yesterday'] = (pd.to_datetime(x, format='%Y%m%d')- timedelta(days=1)) 
df['game_yesterday'] = df.yesterday.isin(df.game_date)
df['game_yesterday'] 
df_after_yes = df.loc[(df.game_yesterday == True)   ].groupby('action_type')

mean_yes = df_after_yes.shot_made_flag.mean()[df_after_yes.shot_made_flag.count() > 10]


df_after_no = df.loc[(df.game_yesterday == False)   ].groupby('action_type')
mean_no = df_after_no.shot_made_flag.mean()[df_after_no.shot_made_flag.count() > 10]

result = pd.concat([mean_yes, mean_no], axis=1, join='inner')

result.columns.values[1] = "Not_Back_To_Back"
result.columns.values[0] = "Back_To_Back"
result.loc[result['Back_To_Back'] + 0.1 < result['Not_Back_To_Back']] 
 



