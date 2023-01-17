import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.metrics import roc_curve, auc
print('Loading data...')

data_path = '../input/'

train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',

                                                'source_system_tab' : 'category',

                                                  'source_screen_name' : 'category',

                                                  'source_type' : 'category',

                                                  'target' : np.uint8,

                                                  'song_id' : 'category'})



songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',

                                                  'language' : 'category',

                                                  'artist_name' : 'category',

                                                  'composer' : 'category',

                                                  'lyricist' : 'category',

                                                  'song_id' : 'category'})

members = pd.read_csv(data_path + 'members.csv',dtype={'city' : 'category',

                                                      'bd' : np.uint8,

                                                      'gender' : 'category',

                                                      'registered_via' : 'category'})

songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
size=train['target'].count()

varidation_data_ratio = 0.15



test=train[int(size*(1-varidation_data_ratio)):size]

train=train[0:int(size*(1-varidation_data_ratio))]
print('Data preprocessing...')

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']

train = train.merge(songs[song_cols], on='song_id', how='left')

test = test.merge(songs[song_cols], on='song_id', how='left')



members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))

members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))

members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))



members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))

members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))

members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))

members = members.drop(['registration_init_time'], axis=1)



def isrc_to_year(isrc):

    if type(isrc) == str:

        if int(isrc[5:7]) > 17:

            return 1900 + int(isrc[5:7])

        else:

            return 2000 + int(isrc[5:7])

    else:

        return np.nan

        

songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)

songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)



train = train.merge(members, on='msno', how='left')

test = test.merge(members, on='msno', how='left')



train = train.merge(songs_extra, on = 'song_id', how = 'left')

test = test.merge(songs_extra, on = 'song_id', how = 'left')



import gc

del members, songs; gc.collect();



for col in train.columns:

    if train[col].dtype == object:

        train[col] = train[col].astype('category')

        test[col] = test[col].astype('category')

X = train.drop(['target'], axis=1)

y = train['target'].values



del train; gc.collect();



d_train = lgb.Dataset(X, y)

watchlist = [d_train]



#Those parameters are almost out of hat, so feel free to play with them. I can tell

#you, that if you do it right, you will get better results for sure ;)

print('Training LGBM model...')

params = {}

params['learning_rate'] = 0.2

params['application'] = 'binary'

params['max_depth'] = 8

params['num_leaves'] = 2**8

params['verbosity'] = 0

params['metric'] = 'auc'



model = lgb.train(params, train_set=d_train, num_boost_round=50, valid_sets=watchlist, \

verbose_eval=5)
X_test = test.drop(['target'], axis=1)

print('Making predictions and Caluculate ROC Area...')

p_test = model.predict(X_test)



fpr, tpr, thresholds = roc_curve(test['target'], p_test)

roc_auc = auc(fpr, tpr)

roc_auc