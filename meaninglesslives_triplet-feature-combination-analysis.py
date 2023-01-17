import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import itertools
from tqdm import tqdm
# https://www.kaggle.com/titericz/the-property-by-giba
feat_old = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', 
            '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 
            'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
            '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 
            'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
            '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
            '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', 
            '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

# https://www.kaggle.com/datagray/another-set-of-ordered-columns
feat_new = ['f3cf9341c','fa11da6df','d47c58fe2','555f18bd3','134ac90df','716e7d74d',
           '1f6b2bafa','174edf08a','5bc7ab64f','a61aa00b0','b2e82c050','26417dec4',
           '51707c671','e8d9394a0','cbbc9c431','6b119d8ce','f296082ec','be2e15279',
           '698d05d29','38e6f8d32','93ca30057','7af000ac2','1fd0a1f2a','41bc25fef',
           '0df1d7b9a','2b2b5187e','bf59c51c3','cfe749e26','ad207f7bb','11114a47a',
           'a8dd5cea5','b88e5de84','1bf8c2597']
    
def get_pred(data,FEATURES, lag=2):
    d1 = data[FEATURES[:-lag]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = data[FEATURES[lag:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[FEATURES[lag - 2]]
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)
train = pd.read_csv('../input/train.csv')
def get_all_pred(data,feat, max_lag):
    target = pd.Series(index=data.index, data=np.zeros(data.shape[0]))
    for lag in range(2, max_lag + 1):
        pred = get_pred(data,feat, lag)
        mask = (target == 0) & (pred != 0)
        target[mask] = pred[mask]
    return target
df = pd.DataFrame()
i = 0
for subset in tqdm(itertools.combinations(feat_old, 3)):
    subset = list(subset)
    pred_train_new = get_all_pred(train,subset,2)
    have_data = pred_train_new != 0
    score = sqrt(mean_squared_error(np.log1p(train.target[have_data]), np.log1p(pred_train_new[have_data])))
    df.loc[i,'subset'] =  str(subset)
    df.loc[i,'score'] =  score
    df.loc[i,'number_predictions'] =  have_data.sum()
#     print(f'Max lag {3}: Score = {sqrt(mean_squared_error(np.log1p(train.target[have_data]), np.log1p(pred_train_new[have_data])))} on {have_data.sum()} out of {train.shape[0]} training samples')
    i = i + 1
df['score'].describe()
sns.distplot(df['score'])
df_sorted = df.sort_values(by='score')
feature_list = df_sorted.loc[:30,'subset'].values
print('Displaying the best 30 triplet subsets')
df_sorted.head(30)
best_feature_list = []
for i in range(len(feature_list)):
    foo = feature_list[i][2:-1].split("'")
    foo.remove(', ')
    foo.remove(', ')
    for feat in foo[:-1]:
        if feat not in best_feature_list:
            best_feature_list.append(feat)
print(best_feature_list)
print('No. of features found based on triplet analysis ',len(best_feature_list))
df_new = pd.DataFrame()
i = 0
for subset in tqdm(itertools.combinations(feat_new, 3)):
    subset = list(subset)
    pred_train_new = get_all_pred(train,subset,2)
    have_data = pred_train_new != 0
    score = sqrt(mean_squared_error(np.log1p(train.target[have_data]), np.log1p(pred_train_new[have_data])))
    df_new.loc[i,'subset'] =  str(subset)
    df_new.loc[i,'score'] =  score
    df_new.loc[i,'number_predictions'] =  have_data.sum()
#     print(f'Max lag {3}: Score = {sqrt(mean_squared_error(np.log1p(train.target[have_data]), np.log1p(pred_train_new[have_data])))} on {have_data.sum()} out of {train.shape[0]} training samples')
    i = i + 1
df_new['score'].describe()
sns.distplot(df_new['score'])
df_sorted_new = df_new.sort_values(by='score')
print('Displaying the best 30 triplet subsets')
df_sorted_new.head(30)
feature_list = df_sorted_new.loc[:30,'subset'].values
best_feature_list_new = []
for i in range(len(feature_list)):
    foo = feature_list[i][2:-1].split("'")
    foo.remove(', ')
    foo.remove(', ')
    for feat in foo[:-1]:
        if feat not in best_feature_list_new:
            best_feature_list_new.append(feat)
print(best_feature_list_new)
print('No. of features found based on triplet analysis ',len(best_feature_list_new))

