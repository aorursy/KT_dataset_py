import numpy as np 
import pandas as pd 
import gc
train = pd.read_csv('../input/train_V2.csv')
train.drop(train.columns.difference(['matchId','groupId','maxPlace','numGroups','winPlacePerc']),axis=1,inplace=True)
init_check = train .groupby(['matchId','groupId'])[['winPlacePerc']].nunique()
init_check.head()
print(init_check[['winPlacePerc']].max())
print('Train # of samples before dropping duplicates {:,}'.format(train.shape[0]))
train.drop_duplicates(subset=['matchId','groupId'],inplace=True)
print('Train # of samples after dropping duplicates {:,}'.format(train.shape[0]))
train_same = train[train['maxPlace'] == train['numGroups']].copy()
train_diff = train[train['maxPlace'] != train['numGroups']].copy()
del train
gc.collect();
print(train_same['winPlacePerc'].isnull().sum())
train_same.dropna(subset = ['winPlacePerc'], inplace = True)
print(train_same['winPlacePerc'].isnull().sum())
print('# of samples for which maxPlace = numGroups: {:,}'.format(train_same.shape[0]))
train_same['winPlaceRank'] = train_same['winPlacePerc'] * (train_same['maxPlace'] - 1)
train_same['winPlaceRank_frac'] = np.mod(np.around(train_same['winPlaceRank'] * 100).astype(int),100)
train_same['winPlaceRank_frac'].value_counts().sort_values()
train_same[ (train_same['winPlaceRank'] > (train_same['maxPlace'] - 1)) | (train_same['winPlaceRank'] < 0) ]
train_diff[train_diff['maxPlace'] < train_diff['numGroups']]
train_diff['numGroups'].min()
train_diff[train_diff['numGroups'] == 1]['winPlacePerc'].unique()
np.sort(train_diff[train_diff['numGroups'] == 1]['maxPlace'].unique())
train_diff[train_diff['numGroups'] == 2]['winPlacePerc'].unique()
np.sort(train_diff[train_diff['numGroups'] == 2]['maxPlace'].unique())
#Renaming train_diff to make for shorter code

train = train_diff.copy()
del train_diff
gc.collect();
train = train[train['numGroups'] > 2]
print('# of samples we are going to examine: {:,}'.format(train.shape[0]))
train.head()
train.sort_values(by=['matchId','winPlacePerc'],inplace=True)

winPlacePerc_gaps = train.groupby('matchId')[['winPlacePerc']].diff()
winPlacePerc_gaps.dropna(inplace=True)
winPlacePerc_gaps = winPlacePerc_gaps.join(train[['matchId','maxPlace','numGroups']],how='left')
winPlacePerc_gaps['winPlacePerc'] = np.around(winPlacePerc_gaps['winPlacePerc'] * 10000).astype(int)

winPlacePerc_gaps_agg = winPlacePerc_gaps.groupby('matchId').agg({'winPlacePerc' : lambda x: set(x), 'maxPlace' : 'mean', 'numGroups' : 'mean'})
winPlacePerc_gaps_agg.head()
winPlacePerc_gaps_agg['winPlacePerc'] = winPlacePerc_gaps_agg['winPlacePerc'].map(lambda x: np.sort(np.array(list(x))))
winPlacePerc_gaps_agg['winPlacePerc'] = 10000 / winPlacePerc_gaps_agg['winPlacePerc']
winPlacePerc_gaps_agg['winPlacePerc'] = winPlacePerc_gaps_agg['winPlacePerc'].map(lambda x: set(np.around(x)))
winPlacePerc_gaps_agg.head()
winPlacePerc_gaps_agg['maxPlace_vote'] = winPlacePerc_gaps_agg.apply(lambda x: (x['maxPlace'] - 1) in x['winPlacePerc'], axis=1)
winPlacePerc_gaps_agg['numGroups_vote'] = winPlacePerc_gaps_agg.apply(lambda x: (x['numGroups'] - 1) in x['winPlacePerc'], axis=1)
print('# of samples we are examining: {:,}'.format(train.shape[0]))
print('# of samples for which maxPlace may not work: {:,}'.format(winPlacePerc_gaps_agg[~winPlacePerc_gaps_agg['maxPlace_vote']].shape[0]))
print('# of samples for which numGroups may not work: {:,}'.format(winPlacePerc_gaps_agg[~winPlacePerc_gaps_agg['numGroups_vote']].shape[0]))
print('# of samples for which both maxPlace and numGroups work: {:,}'.format(winPlacePerc_gaps_agg[winPlacePerc_gaps_agg['numGroups_vote'] & winPlacePerc_gaps_agg['maxPlace_vote']].shape[0]))
winPlacePerc_gaps_agg[~winPlacePerc_gaps_agg['maxPlace_vote']]
train[train['matchId'] == '668560ba6622c2']
train[train['numGroups'] == 3].sort_values(by=['matchId','winPlacePerc'])
test = pd.read_csv('../input/test_V2.csv')
print('# of samples in the test set: {:,}'.format(test.shape[0]))
print('# of samples in the test set for which numGroups is less than or equal to 3: {:,}'.format(test[test['numGroups'] <= 3].shape[0]))
test[test['numGroups'] <=3].groupby('numGroups')['groupId'].size()