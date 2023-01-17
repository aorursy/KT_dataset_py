import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

shots = pd.read_csv('../input/shot_logs.csv')

shots.head()


def clock_to_seconds(t):
    m, s = [int(i) for i in t.split(':')]
    return 60*m + s

def miss_or_make(t):
    if t=="missed":
        return 0
    else:
        return 1
    
shots['GAME_CLOCK']=shots.GAME_CLOCK.apply(clock_to_seconds)
shots['SHOT_RESULT']=shots.SHOT_RESULT.apply(miss_or_make)
shots['TOUCH_TIME'].sort_values(ascending=True).head()


ttimefilter=shots['TOUCH_TIME']>=0

shots=shots[ttimefilter]
shots['TOUCH_TIME'].sort_values(ascending=True).head()
def range_cat(shot_type, t):
    
    if shot_type==3:
        return 'three'
    elif t < 6:
        return 'close'
    elif (t>=6) & (t<11):
        return 'mid_short'
    elif (t>=11) & (t<17):
        return 'mid'
    elif (t>=17):
        return 'mid_long'
    else:
        return 'non_measured'
    
shots['DIST_CATEGORY']=shots.apply(lambda x: range_cat(x['PTS_TYPE'], x['SHOT_DIST']), axis=1)



players= pd.concat([shots['player_id'], shots['player_name']], axis=1, keys=['PLAYER_ID', 'PLAYER'])
players=players.drop_duplicates()
players.head()



for index, row in players.iterrows():
    player_name=row['PLAYER_ID']
    
    players.loc[(players['PLAYER_ID'])==player_name, 'FGM']=\
    shots[ (shots['player_id']==player_name) & (shots['SHOT_RESULT']==True) ]['player_id'].count() 
    
    players.loc[(players['PLAYER_ID'])==player_name, 'FGA']=\
    shots[ (shots['player_id']==player_name) ]['player_id'].count()
    
    players.loc[(players['PLAYER_ID'])==player_name, '3PM']=\
    shots[ (shots['player_id']==player_name) & (shots['PTS_TYPE']==3) & (shots['SHOT_RESULT']==True) ]['player_id'].count()
    
    
    players.loc[(players['PLAYER_ID'])==player_name, 'TOUCH TIME']=\
    shots[ (shots['player_id']==player_name) & (shots['TOUCH_TIME']) ]['TOUCH_TIME'].mean() 
    
    players.loc[(players['PLAYER_ID'])==player_name, 'DRIBBLES']=\
    shots[ (shots['player_id']==player_name) & (shots['DRIBBLES']) ]['DRIBBLES'].mean() 
    
    players.loc[(players['PLAYER_ID'])==player_name, 'FG%']=\
    players['FGM']/players['FGA']
    
    #Adding a percentage column for each players shot distance type. 
    #There's probably a way more efficient way to do this but I haven't figured it out yet
    players.loc[(players['PLAYER_ID'])==player_name, 'close_total']= int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='close')]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'close_made']=int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='close')& (shots['SHOT_RESULT']==True)]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'closeFG%']=players['close_made']/players['close_total']


    players.loc[(players['PLAYER_ID'])==player_name, 'mid_short_total']= int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='mid_short')]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'mid_short_made']=int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='mid_short')& (shots['SHOT_RESULT']==True)]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'mid_shortFG%']=players['mid_short_made']/players['mid_short_total']

    
    players.loc[(players['PLAYER_ID'])==player_name, 'mid_total']= int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='mid')]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'mid_made']=int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='mid')& (shots['SHOT_RESULT']==True)]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'midFG%']=players['mid_made']/players['mid_total']

    
    players.loc[(players['PLAYER_ID'])==player_name, 'three_total']= int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='three')]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'three_made']=int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='three')& (shots['SHOT_RESULT']==True)]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'threeFG%']=players['three_made']/players['three_total']

    players.loc[(players['PLAYER_ID'])==player_name, 'mid_long_total']= int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='mid_long')]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'mid_long_made']=int(shots[(shots['player_id']==player_name)&(shots['DIST_CATEGORY']=='mid_long')& (shots['SHOT_RESULT']==True)]['player_id'].count())
    players.loc[(players['PLAYER_ID'])==player_name, 'mid_longFG%']=players['mid_long_made']/players['mid_long_total']

   
players.head()
players=players.rename(index=str,columns={'PLAYER_ID':'player_id'})
merged=shots.merge(players, how='left', on='player_id')
merged.loc[(merged['DIST_CATEGORY']=='mid'), 'playerBRK%']=merged['midFG%']
merged.loc[(merged['DIST_CATEGORY']=='mid_short'), 'playerBRK%']=merged['mid_shortFG%']
merged.loc[(merged['DIST_CATEGORY']=='mid_long'), 'playerBRK%']=merged['mid_longFG%']
merged.loc[(merged['DIST_CATEGORY']=='close'), 'playerBRK%']=merged['closeFG%']
merged.loc[(merged['DIST_CATEGORY']=='three'), 'playerBRK%']=merged['threeFG%']
merged=merged.reset_index(drop=True)
merged.loc[:,['PLAYER','DIST_CATEGORY','midFG%','mid_shortFG%','closeFG%','mid_longFG%','threeFG%','playerBRK%']].sample(10)
columns_to_keep=shots.columns.values
columns_to_keep=np.append(columns_to_keep,'playerBRK%')
columns_to_keep=np.array_repr(columns_to_keep).replace('\n','')
columns_to_keep
knn=KNeighborsRegressor(algorithm='kd_tree',leaf_size=300, n_neighbors=50)

feature_cols=['playerBRK%', 'CLOSE_DEF_DIST', 'TOUCH_TIME','PERIOD',]
train_df = merged.copy().iloc[:int(len(merged)*.75)]
test_df = merged.copy().iloc[int(len(merged)*.75):]

test1=train_df[feature_cols]
test2=train_df['SHOT_RESULT']


knn.fit(test1, test2)
predictions = knn.predict(test_df[feature_cols])

predictions
test_df['predictions']=predictions

test_df.loc[:,['player_name', 'player_id','playerBRK%', 'CLOSE_DEF_DIST', 'TOUCH_TIME','PERIOD','DIST_CATEGORY','SHOT_RESULT','predictions']].sample(10)
