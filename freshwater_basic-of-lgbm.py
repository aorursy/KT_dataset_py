# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print('Loading data...')

data_path = '../input/'

train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',

                                                'source_system_tab' : 'category',

                                                  'source_screen_name' : 'category',

                                                  'source_type' : 'category',

                                                  'target' : np.uint8,

                                                  'song_id' : 'category'})

test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',

                                                'source_system_tab' : 'category',

                                                'source_screen_name' : 'category',

                                                'source_type' : 'category',

                                                'song_id' : 'category'})

songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',

                                                  'language' : 'category',

                                                  'artist_name' : 'category',

                                                  'composer' : 'category',

                                                  'lyricist' : 'category',

                                                  'song_id' : 'category'})

members = pd.read_csv(data_path + 'members.csv',

                      dtype={'city' : 'category',

                             'bd' : np.uint8,

                             'gender' : 'category',

                             'registered_via' : 'category'}

                     ,parse_dates=["registration_init_time","expiration_date"])

songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')



print('Data preprocessing...')
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']

train = train.merge(songs[song_cols], on='song_id', how='left')

test = test.merge(songs[song_cols], on='song_id', how='left')



members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))

#members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))

#members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))



members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))

members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))

#members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))



# exepting some unimportanat features





# Convert date to number of days

members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days.astype(int)



#members = members.drop(['registration_init_time'], axis=1)

#members = members.drop(['expiration_date'], axis=1)



# categorize membership_days 

members['membership_days'] = members['membership_days']//200

members['membership_days'] = members['membership_days'].astype('category')
member_cols = ['msno','city','registered_via', 'registration_year', 'expiration_year', 'membership_days']



train = train.merge(members[member_cols], on='msno', how='left')

test = test.merge(members[member_cols], on='msno', how='left')
train.info()
def isrc_to_year(isrc):

    if type(isrc) == str:

        if int(isrc[5:7]) > 17:

            return int(isrc[5:7])//5

        else:

            return int(isrc[5:7])//5

    else:

        return np.nan

#categorize song_year per 5years



songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)

songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)
train = train.merge(songs_extra, on = 'song_id', how = 'left')

test = test.merge(songs_extra, on = 'song_id', how = 'left')
# use only first vector of genre

train['genre_ids'] = train['genre_ids'].str.split('|').str[0]
temp_song_length = train['song_length']
train.drop('song_length', axis = 1, inplace = True)

test.drop('song_length',axis = 1 , inplace =True)
train.head()
#

song_count = train.loc[:,["song_id","target"]]



# measure repeat count by played songs

song_count1 = song_count.groupby(["song_id"],as_index=False).sum().rename(columns={"target":"repeat_count"})



# count play count by songs

song_count2 = song_count.groupby(["song_id"],as_index=False).count().rename(columns = {"target":"play_count"})
song_repeat = song_count1.merge(song_count2,how="inner",on="song_id")

song_repeat["repeat_percentage"] = round((song_repeat['repeat_count']*100) / song_repeat['play_count'],1)

song_repeat['repeat_count'] = song_repeat['repeat_count'].astype('int')

song_repeat['repeat_percentage'] = song_repeat['repeat_percentage'].replace(100.0,np.nan)

#cuz most of 100.0 are played=1 repeated=1 values. I think it is not fair compare with other played a lot songs
train = train.merge(song_repeat,on="song_id",how="left")

test = test.merge(song_repeat,on="song_id",how="left")
# type cast

test['song_id'] = test['song_id'].astype('category')

test['repeat_count'] = test['repeat_count'].fillna(0)

test['repeat_count'] = test['repeat_count'].astype('int')

test['play_count'] = test['play_count'].fillna(0)

test['play_count'] = test['play_count'].astype('int')
#train['repeat_percentage'].replace(100.0,np.nan)

#test['repeat_percentage'].replace(100.0,np.nan)

#train['repeat_count'].replace(0,np.nan)

#test['repeat_count'].replace(0,np.nan)

#train['play_count'].replace(0,np.nan)

#test['play_count'].replace(0,np.nan)



#train
artist_count = train.loc[:,["artist_name","target"]]



# measure repeat count by played songs

artist_count1 = artist_count.groupby(["artist_name"],as_index=False).sum().rename(columns={"target":"repeat_count_artist"})



# measure play count by songs

artist_count2 = artist_count.groupby(["artist_name"],as_index=False).count().rename(columns = {"target":"play_count_artist"})



artist_repeat = artist_count1.merge(artist_count2,how="inner",on="artist_name")
artist_repeat["repeat_percentage_artist"] = round((artist_repeat['repeat_count_artist']*100) / artist_repeat['play_count_artist'],1)

artist_repeat['repeat_count_artist'] = artist_repeat['repeat_count_artist'].fillna(0)

artist_repeat['repeat_count_artist'] = artist_repeat['repeat_count_artist'].astype('int')

artist_repeat['repeat_percentage_artist'] = artist_repeat['repeat_percentage_artist'].replace(100.0,np.nan)
#use only repeat_percentage_artist

del artist_repeat['repeat_count_artist']

#del artist_repeat['play_count_artist']
#merge it with artist_name to train dataframe

train = train.merge(artist_repeat,on="artist_name",how="left")

test = test.merge(artist_repeat,on="artist_name",how="left")
del train['artist_name']

del test['artist_name']
msno_count = train.loc[:,["msno","target"]]



# count repeat count by played songs

msno_count1 = msno_count.groupby(["msno"],as_index=False).sum().rename(columns={"target":"repeat_count_msno"})



# count play count by songs

msno_count2 = msno_count.groupby(["msno"],as_index=False).count().rename(columns = {"target":"play_count_msno"})



msno_repeat = msno_count1.merge(msno_count2,how="inner",on="msno")
msno_repeat["repeat_percentage_msno"] = round((msno_repeat['repeat_count_msno']*100) / msno_repeat['play_count_msno'],1)

msno_repeat['repeat_count_msno'] = msno_repeat['repeat_count_msno'].fillna(0)

msno_repeat['repeat_count_msno'] = msno_repeat['repeat_count_msno'].astype('int')

#msno_repeat['repeat_percentage_msno'] = msno_repeat['repeat_percentage_msno'].replace(100.0,np.nan)

# it can be meaningful so do not erase 100.0 
#merge it with msno to train dataframe

train = train.merge(msno_repeat,on="msno",how="left")

test = test.merge(msno_repeat,on="msno",how="left")
#del train['msno']

#del test['msno']
import gc

#del members, songs; gc.collect();



for col in train.columns:

    if train[col].dtype == object:

        train[col] = train[col].astype('category')

        test[col] = test[col].astype('category')
train['song_year'] = train['song_year'].astype('category')

test['song_year'] = test['song_year'].astype('category')
train.head()
train.info()
drop_list = ['repeat_count','repeat_percentage',

             'repeat_percentage_artist',

             'repeat_count_msno','repeat_percentage_msno'

            ]

train = train.drop(drop_list,axis=1)

test = test.drop(drop_list,axis=1)
test
test['play_count_msno'] = test['play_count_msno'].fillna(0)

test['play_count_msno'] = test['play_count_msno'].astype('int')





train['play_count_artist'] = train['play_count_artist'].fillna(0)

test['play_count_artist'] = test['play_count_artist'].fillna(0)

train['play_count_artist'] = train['play_count_artist'].astype('int')

test['play_count_artist'] = test['play_count_artist'].astype('int')

from sklearn.model_selection import KFold

# Create a Cross Validation with 3 splits

kf = KFold(n_splits=3)



predictions = np.zeros(shape=[len(test)])



# For each KFold

for train_indices ,validate_indices in kf.split(train) : 

    train_data = lgb.Dataset(train.drop(['target'],axis=1).loc[train_indices,:],label=train.loc[train_indices,'target'])

    val_data = lgb.Dataset(train.drop(['target'],axis=1).loc[validate_indices,:],label=train.loc[validate_indices,'target'])



    params = {

            'objective': 'binary',

            'boosting': 'gbdt',

            'learning_rate': 0.2 ,

            'verbose': 0,

            'num_leaves': 2**8,

            'bagging_fraction': 0.95,

            'bagging_freq': 1,

            'bagging_seed': 1,

            'feature_fraction': 0.9,

            'feature_fraction_seed': 1,

            'max_bin': 256,

            'num_rounds': 80,

            'metric' : 'auc'

        }    

    # Train the model    

    lgbm_model = lgb.train(params, train_data, 100, valid_sets=[val_data])

    predictions += lgbm_model.predict(test.drop(['id'],axis=1))

    del lgbm_model

    # We get the ammount of predictions from the prediction list, by dividing the predictions by the number of Kfolds.

predictions = predictions/3



INPUT_DATA_PATH = '../input/'



# Read the sample_submission CSV

submission = pd.read_csv(INPUT_DATA_PATH + '/sample_submission.csv')

# Set the target to our predictions

submission.target=predictions

# Save the submission file

submission.to_csv('submission.csv',index=False)