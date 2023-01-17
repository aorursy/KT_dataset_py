import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import scipy 

# import data
train=pd.read_csv('../input/train_V2.csv',sep=',',header=0)

# drop na values
train=train.dropna()
## drop max(kills) > numplayers games
# calculate number of players
train['num']=train.groupby('matchId')['Id'].transform('count')
# calculate max kills in a game
train['max']=train.groupby('matchId')['kills'].transform('max')
#dropping
train=train[train['num']>train['max']]
# sorting
# train[train.matchId=='a10357fd1a4a91'].sort_values(['groupId'])
# drop maxkills and numply columns
train=train.drop(['num','max'],axis=1)
group_rank = train[['matchId','groupId','winPlacePerc','maxPlace','numGroups']].drop_duplicates()

group_rank['rank'] = group_rank.groupby('matchId')['winPlacePerc'].rank(method='min',ascending=False)

group_rank['gap'] = 1 / (group_rank['numGroups'] - 1)

group_rank.loc[group_rank['maxPlace']==1,'re_rank'] = 0

group_rank.loc[group_rank['maxPlace']!=1,'re_rank'
              ] = group_rank['gap'] * (group_rank['numGroups'] - group_rank['rank'])

train = pd.merge(train, group_rank[['matchId','groupId','re_rank']], how='left', on=['matchId','groupId'])
# Decompose matchType
train.loc[train.matchType.str.contains('fpp'),'fpp'] = 1
train.loc[~train.matchType.str.contains('fpp'),'fpp'] = 0
train.loc[train.matchType.str.contains('normal'),'normal'] = 1
train.loc[~train.matchType.str.contains('normal'),'normal'] = 0
train.loc[train.matchType.str.contains('squad'),'squad'] = 1
train.loc[~train.matchType.str.contains('squad'),'squad'] = 0
train.loc[train.matchType.str.contains('duo'),'duo'] = 1
train.loc[~train.matchType.str.contains('duo'),'duo'] = 0
train.loc[train.matchType.str.contains('solo'),'solo'] = 1
train.loc[~train.matchType.str.contains('solo'),'solo'] = 0
# create group- and match-level kills stats
train['maxGrp_killPoints'] = train.groupby('groupId')['killPoints'].transform('max')

train['Perc_kills'] =  train['kills']/train.groupby('matchId')['kills'].transform('sum')
train['maxPerc_kills'] = train.groupby('matchId')['Perc_kills'].transform('max')
#train['minPerc_kills'] = train.groupby('matchId')['Perc_kills'].transform('min')
train['Rank_kills'] = train.groupby('matchId')['kills'].rank(method='min',ascending=False,pct=False)
train['maxRank_kills'] = train.groupby('matchId')['Rank_kills'].transform('max')
#train['minRank_kills'] = train.groupby('matchId')['Rank_kills'].transform('min')
train['max_kills'] = train.groupby('matchId')['kills'].transform('max')
train['kill_Death'] = train['max_kills']/train.groupby('matchId')['Id'].transform('count')
train['RankPerc_kills'] = train.groupby('matchId')['kills'].rank(method='max',ascending=True,pct=True)
train['tkills']=train.groupby('groupId')['kills'].transform('sum')
# create group- and match-level stats for damageDealt

train['Perc_damageDt'] =  train['damageDealt']/train.groupby('matchId')['damageDealt'].transform('sum')
train['maxPerc_damageDt'] = train.groupby('matchId')['Perc_damageDt'].transform('max')
#train['minPerc_damageDt'] = train.groupby('matchId')['Perc_damageDt'].transform('min')
train['Rank_damageDt'] = train.groupby('matchId')['damageDealt'].rank(method='min',ascending=False,pct=False)
train['maxRank_damageDt'] = train.groupby('matchId')['Rank_damageDt'].transform('max')
#train['minRank_damageDt'] = train.groupby('matchId')['Rank_damageDt'].transform('min')
train['max_damageDt'] = train.groupby('matchId')['damageDealt'].transform('max')
# walkDistance

train['walkDistance']=np.log(train['walkDistance']+1)
train['Perc_wkdis'] =  train['walkDistance']/train.groupby('matchId')['walkDistance'].transform('sum')
train['maxPerc_wkdis'] = train.groupby('matchId')['Perc_wkdis'].transform('max')
#train['minPerc_wkdis'] = train.groupby('matchId')['Perc_wkdis'].transform('min')
train['Rank_wkdis'] = train.groupby('matchId')['walkDistance'].rank(method='min',ascending=False,pct=False)
train['maxRank_wkdis'] = train.groupby('matchId')['Rank_wkdis'].transform('max')
#train['minRank_wkdis'] = train.groupby('matchId')['Rank_wkdis'].transform('min')
train['max_wkdis'] = train.groupby('matchId')['walkDistance'].transform('max')
#train['RankPerc_wkdis'] = train.groupby('matchId')['walkDistance'].rank(method='max',ascending=True,pct=True)
# create a new column indicate the max(walkDistance) of the group
train['maxwkdis']=train.groupby('groupId')['walkDistance'].transform('max') 


# rideDistance

train['Perc_ridedis'] =  train['rideDistance']/train.groupby('matchId')['rideDistance'].transform('sum')
train['maxPerc_ridedis'] = train.groupby('matchId')['Perc_ridedis'].transform('max')
#train['minPerc_ridedis'] = train.groupby('matchId')['Perc_ridedis'].transform('min')
train['Rank_ridedis'] = train.groupby('matchId')['rideDistance'].rank(method='min',ascending=False,pct=False)
train['maxRank_ridedis'] = train.groupby('matchId')['Rank_ridedis'].transform('max')
#train['minRank_ridedis'] = train.groupby('matchId')['Rank_ridedis'].transform('min')
#train['RankPerc_ridedis'] = train.groupby('matchId')['rideDistance'].rank(method='max',ascending=True,pct=True)
train['max_ridedis'] = train.groupby('matchId')['rideDistance'].transform('max')
train['maxGrp_ridedis']=train.groupby('groupId')['rideDistance'].transform('max') 
# boosts, heals, combine assists and revives
train['Perc_boo'] =  train['boosts']/train.groupby('matchId')['boosts'].transform('sum')
#train['maxPerc_boo'] = train.groupby('matchId')['Perc_boo'].transform('max')
#train['minPerc_boo'] = train.groupby('matchId')['Perc_boo'].transform('min')
train['max_boo'] = train.groupby('matchId')['boosts'].transform('max')

train['Perc_heals'] =  train['heals']/train.groupby('matchId')['heals'].transform('sum')
#train['maxPerc_heals'] = train.groupby('matchId')['Perc_heals'].transform('max')
#train['minPerc_heals'] = train.groupby('matchId')['Perc_heals'].transform('min')
train['max_heals'] = train.groupby('matchId')['heals'].transform('max')

# combine variables
train['assReviv'] = train['assists']+train['revives']
train = train.drop(columns=['assists','revives'],axis=1)

train['Perc_assReviv'] =  train['assReviv']/train.groupby('matchId')['assReviv'].transform('sum')
#train['maxPerc_revives'] = train.groupby('matchId')['Perc_revives'].transform('max')
#train['minPerc_revives'] = train.groupby('matchId')['Perc_revives'].transform('min')
train['max_assReviv'] = train.groupby('matchId')['assReviv'].transform('max')
## configuring features
# create new game stats
train['teamSize'] = train.groupby('groupId')['Id'].transform('count')
train['numEnemies'] = train.groupby('matchId')['Id'].transform('count')-train['teamSize']
train['pace'] = train.groupby('matchId')['kills'].transform('sum')/train['matchDuration']
train = train.drop('matchType',axis = 1)
train = train.drop(columns=['vehicleDestroys','roadKills'],axis=1)
# split training and validation sets

match_ids = train['matchId'].drop_duplicates()

msk=np.random.rand(len(match_ids))<0.8
valid = train[train['matchId'].isin(match_ids[~msk])]
train = train[train['matchId'].isin(match_ids[msk])]
# sort train_aggtrain by winPlacePerc 
train = train.sort_values(by=['matchId','re_rank'])

# creat y array
y_train = train['re_rank'].values

# create group list, a list that contains the number of rows in each group, should be the same order as in train set?
group_train = train.groupby('matchId',as_index=False)['Id'].agg('count')['Id'].tolist()

# convert df to csr_matrix
x_train = scipy.sparse.csr_matrix(train.drop(columns=['matchId','groupId','Id','re_rank','winPlacePerc'],axis=1))
# sort train_aggtrain by winPlacePerc 
valid = valid.sort_values(by=['matchId','re_rank'])

# creat y array
y_valid = valid['re_rank'].values

# create group list, a list that contains the number of rows in each group, should be the same order as in train set?
group_valid = valid.groupby('matchId',as_index=False)['Id'].agg('count')['Id'].tolist()

# convert df to csr_matrix
x_valid = scipy.sparse.csr_matrix(valid.drop(columns=['matchId','groupId','Id','re_rank','winPlacePerc'],axis=1))
# import testing set
test=pd.read_csv('../input/test_V2.csv',sep=',',header=0)

## configuring features

# Decompose matchType

test.loc[test.matchType.str.contains('fpp'),'fpp'] = 1
test.loc[~test.matchType.str.contains('fpp'),'fpp'] = 0
test.loc[test.matchType.str.contains('normal'),'normal'] = 1
test.loc[~test.matchType.str.contains('normal'),'normal'] = 0
test.loc[test.matchType.str.contains('squad'),'squad'] = 1
test.loc[~test.matchType.str.contains('squad'),'squad'] = 0
test.loc[test.matchType.str.contains('duo'),'duo'] = 1
test.loc[~test.matchType.str.contains('duo'),'duo'] = 0
test.loc[test.matchType.str.contains('solo'),'solo'] = 1
test.loc[~test.matchType.str.contains('solo'),'solo'] = 0
# create group- and match-level kills stats

test['maxGrp_killPoints'] = test.groupby('groupId')['killPoints'].transform('max')
test['Perc_kills'] =  test['kills']/test.groupby('matchId')['kills'].transform('sum')
test['maxPerc_kills'] = test.groupby('matchId')['Perc_kills'].transform('max')
test['Rank_kills'] = test.groupby('matchId')['kills'].rank(method='min',ascending=False,pct=False)
test['maxRank_kills'] = test.groupby('matchId')['Rank_kills'].transform('max')
test['max_kills'] = test.groupby('matchId')['kills'].transform('max')
test['kill_Death'] = test['max_kills']/test.groupby('matchId')['Id'].transform('count')
test['RankPerc_kills'] = test.groupby('matchId')['kills'].rank(method='max',ascending=True,pct=True)
test['tkills']=test.groupby('groupId')['kills'].transform('sum')

# create group- and match-level stats for damageDealt

test['Perc_damageDt'] =  test['damageDealt']/test.groupby('matchId')['damageDealt'].transform('sum')
test['maxPerc_damageDt'] = test.groupby('matchId')['Perc_damageDt'].transform('max')
test['Rank_damageDt'] = test.groupby('matchId')['damageDealt'].rank(method='min',ascending=False,pct=False)
test['maxRank_damageDt'] = test.groupby('matchId')['Rank_damageDt'].transform('max')
test['max_damageDt'] = test.groupby('matchId')['damageDealt'].transform('max')


# walkDistance

test['walkDistance']=np.log(test['walkDistance']+1)
test['Perc_wkdis'] =  test['walkDistance']/test.groupby('matchId')['walkDistance'].transform('sum')
test['maxPerc_wkdis'] = test.groupby('matchId')['Perc_wkdis'].transform('max')
test['Rank_wkdis'] = test.groupby('matchId')['walkDistance'].rank(method='min',ascending=False,pct=False)
test['maxRank_wkdis'] = test.groupby('matchId')['Rank_wkdis'].transform('max')
test['max_wkdis'] = test.groupby('matchId')['walkDistance'].transform('max')
test['maxwkdis']=test.groupby('groupId')['walkDistance'].transform('max') 


# rideDistance

test['Perc_ridedis'] =  test['rideDistance']/test.groupby('matchId')['rideDistance'].transform('sum')
test['maxPerc_ridedis'] = test.groupby('matchId')['Perc_ridedis'].transform('max')
test['Rank_ridedis'] = test.groupby('matchId')['rideDistance'].rank(method='min',ascending=False,pct=False)
test['maxRank_ridedis'] = test.groupby('matchId')['Rank_ridedis'].transform('max')
test['max_ridedis'] = test.groupby('matchId')['rideDistance'].transform('max')
test['maxGrp_ridedis']=test.groupby('groupId')['rideDistance'].transform('max') 

# boosts

test['Perc_boo'] =  test['boosts']/test.groupby('matchId')['boosts'].transform('sum')
test['max_boo'] = test.groupby('matchId')['boosts'].transform('max')

# heals

test['Perc_heals'] =  test['heals']/test.groupby('matchId')['heals'].transform('sum')
test['max_heals'] = test.groupby('matchId')['heals'].transform('max')

# combine assists and revives

test['assReviv'] = test['assists']+test['revives']
test = test.drop(columns=['assists','revives'],axis=1)

test['Perc_assReviv'] =  test['assReviv']/test.groupby('matchId')['assReviv'].transform('sum')
test['max_assReviv'] = test.groupby('matchId')['assReviv'].transform('max')

# create new game stats

test['teamSize'] = test.groupby('groupId')['Id'].transform('count')
test['numEnemies'] = test.groupby('matchId')['Id'].transform('count')-test['teamSize']
test['pace'] = test.groupby('matchId')['kills'].transform('sum')/test['matchDuration']

# drop low importance columns

test = test.drop('matchType',axis = 1)
test = test.drop(columns=['vehicleDestroys','roadKills'],axis=1)

# sort
test = test.sort_values(by=['matchId'])

# create group list
group_test = test.groupby('matchId',as_index=False)['Id'].agg('count')['Id'].tolist()

# convert to csr_matrix
x_test = scipy.sparse.csr_matrix(test.drop(axis=1,columns=['Id','groupId','matchId']))
## modeling and predicting
## n_estimate,max_depth configured
params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
          'gamma': 0, 'min_child_weight': 60,
          'max_depth': 14, 'n_estimators': 6,
          'colsample_bytree': 1,'subsample': 1,
          'eval_metric': 'mae'}

model = xgb.sklearn.XGBRanker(**params)
model.fit(x_train, y_train, group_train,
          eval_set=[(x_valid, y_valid)], eval_group=[group_valid])

## prediction
pred = model.predict(x_test)
# adjust prediction by group based on maxPlace

# take the average of a group's percentile prediction as the group's placement
test['pred'] = pred
test['mean_pred'] = test.groupby('groupId')['pred'].transform('mean')

# create a group-level dataframe
group_rank = test[['matchId','groupId','mean_pred','maxPlace','numGroups']].drop_duplicates()

# create integer ranking based on percentile prediction
group_rank['prank'] = group_rank.groupby('matchId')['mean_pred'].rank(method='min',ascending=False)
# number of empty placement due to inconsistency between numGroups and maxPlace
group_rank['emp_ranks'] = group_rank['maxPlace'] - group_rank['numGroups']

# gap between each percentile placement
group_rank['gap'] = 1 / (group_rank['maxPlace'] - 1)

## calculate ranking off of maxPlace

# if maxPlace==1, then all rank 0
group_rank.loc[group_rank['maxPlace']==1,'winPlacePerc'] = 0

# maxPlace!=1
group_rank.loc[(group_rank['maxPlace']!=1) & (group_rank['prank']<(group_rank['numGroups']/2).astype(int)),'winPlacePerc'
              ] = group_rank['gap'] * (group_rank['maxPlace'] - group_rank['prank'])
group_rank.loc[(group_rank['maxPlace']!=1) & (group_rank['prank']>(group_rank['numGroups']/2).astype(int)),'winPlacePerc'
              ] = group_rank['gap'] * (group_rank['maxPlace'] - group_rank['prank'] - group_rank['emp_ranks'])
group_rank.loc[(group_rank['maxPlace']!=1) & (group_rank['prank']==(group_rank['numGroups']/2).astype(int)),'winPlacePerc'
              ] = group_rank['gap'] * (group_rank['maxPlace'] - group_rank['prank'] - (group_rank['emp_ranks']/2).astype(int))
# assign winPlacePerc back to individual player
test = pd.merge(test, group_rank[['matchId','groupId','winPlacePerc']], on=['matchId','groupId'], how='left')
test[['Id','winPlacePerc']].to_csv('submission2_0120.csv',index=False, header=True)