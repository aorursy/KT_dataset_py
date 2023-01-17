# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import lightgbm as lgb

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

FOLDER = "../input/"

#print(check_output(["ls", FOLDER]).decode("utf8"))
print("[r] songs_extra")



    

songs_extra = pd.read_csv(FOLDER + 'song_extra_info.csv')
def isrc_to_country(isrc):

    

    if type(isrc) == str:

        return isrc[:2]

    else:

        return np.nan



def isrc_to_year(isrc):

    if type(isrc) == str:

        if int(isrc[5:7]) > 17:

            return 117 - int(isrc[5:7])

        else:

            return 17 - int(isrc[5:7])

    else:

        return np.nan



def is_english(name):

    if(type(name) != str):

        print("not str")

        return 0

    name = name.lower()

    if len(name) > 0 and name[0] >= 'a' and name[0] <= 'z':

        return 1

    return 0
# songs_extra['song_country'] = songs_extra['isrc'].apply(isrc_to_country)

# songs_extra.song_country.fillna('unknown', inplace=True)

# songs_extra['song_country'] = songs_extra['song_country'].astype('category')



songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)

songs_extra['is_english'] = songs_extra['name'].apply(is_english)



songs_extra.is_english = songs_extra.is_english.astype('category')



songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import KFold



print("[r] train")

df_train = pd.read_csv(FOLDER + 'train.csv',dtype={'msno' : 'category',

													'source_system_tab' : 'category',

                                                  'source_screen_name' : 'category',

                                                  'source_type' : 'category',

                                                  'target' : np.uint8,

                                                  'song_id' : 'category'})





print("[r] test")

df_test = pd.read_csv(FOLDER + 'test.csv',dtype={'msno' : 'category',

												'source_system_tab' : 'category',

                                                  'source_screen_name' : 'category',

                                                  'source_type' : 'category',

                                                'song_id' : 'category'})



m1 = df_train.msno.unique()

m2 = df_test.msno.unique()

m3 = df_test.msno.append(df_test.msno).unique()

print(len(m1), len(m2), len(m3))



print("[r] members")

df_members = pd.read_csv(FOLDER + 'members.csv',dtype={'city' : 'category',

                                                      'bd' : 'category',

                                                      'gender' : 'category',

                                                      'registered_via' : 'category'})





print("[f] members")





print("[m] test,members")

df_test = pd.merge(left = df_test,right = df_members,how='left',on='msno')





print("[m] train,members")

df_train = pd.merge(left = df_train,right = df_members,how='left',on='msno')





del df_members



print("[r] songs")

df_songs = pd.read_csv(FOLDER + 'songs.csv',dtype={'genre_ids': 'category',

                                                  'language' : 'category',

                                                  'artist_name' : 'category',

                                                  'composer' : 'category',

                                                  'lyricist' : 'category',

                                                  'song_id' : 'category'})



# def is_17_or_45(lang):

#     if '17' in str(lang) or '45' in str(lang):

#         return 1

#     return 0





# df_songs['is_17_45'] = df_songs.language.apply(is_17_or_45)



# print("[count] genre_ids")



# genre_count_dict = df_songs.genre_ids.value_counts().to_dict()



# def get_genre_count(genre):

#     return genre_count_dict[genre]



# df_songs['genre_count'] = df_songs.genre_ids.apply(get_genre_count)



print("[m] songs,test")

df_test = pd.merge(left = df_test,right = df_songs,how = 'left',on='song_id')

df_test.song_length.fillna(200000,inplace=True)

df_test.song_length = df_test.song_length.astype(np.uint32)

# df_test.song_id = df_test.song_id.astype('category')



print("[m] train,songs")

df_train = pd.merge(left = df_train,right = df_songs,how = 'left',on='song_id')

df_train.song_length.fillna(200000,inplace=True)

df_train.song_length = df_train.song_length.astype(np.uint32)

# df_train.song_id = df_train.song_id.astype('category')









print("[m] songs_extra,train")

df_train = df_train.merge(songs_extra, on = 'song_id', how = 'left')

df_test = df_test.merge(songs_extra, on = 'song_id', how = 'left')

df_train.song_id = df_train.song_id.astype('category')

df_test.song_id = df_test.song_id.astype('category')

del df_songs



# print("[m] churn,train")

# df_train = df_train.merge(churn, on = 'msno', how = 'left')

# df_test = df_test.merge(churn, on = 'msno', how = 'left')



# number of times a song has been played before

# print("[count] songs played")

# _dict_count_song_played_train = {k: v for k, v in df_train['song_id'].value_counts().iteritems()}

# _dict_count_song_played_test = {k: v for k, v in df_test['song_id'].value_counts().iteritems()}

# def count_song_played(x):

#     try:

#         return _dict_count_song_played_train[x]

#     except KeyError:

#         try:

#             return _dict_count_song_played_test[x]

#         except KeyError:

#             return 0

    



# df_train['count_song_played'] = df_train['song_id'].apply(count_song_played).astype(np.int64)

# df_test['count_song_played'] = df_test['song_id'].apply(count_song_played).astype(np.int64)





# df_train.is_churn = df_train.is_churn.astype('category')

# df_test.is_churn = df_test.is_churn.astype('category')





df_train.msno = df_train.msno.astype('category')

df_test.msno = df_test.msno.astype('category')



SPLITS = 5

# kf = KFold(n_splits=SPLITS)



kf = KFold(n_splits=SPLITS)



predictions = np.zeros(shape=[len(df_test)])
df_train.song_id = df_train.song_id.astype('category')

df_test.song_id = df_test.song_id.astype('category')



# df_test['song_minutes'] = df_test.song_length.apply(lambda x: int(x/1000/60.0))

# df_train['song_minutes'] = df_train.song_length.apply(lambda x: int(x/1000/60.0))

SPLITS = 4

kf = KFold(n_splits=SPLITS)

print("[count] composer")

composers = df_train.composer.append(df_test.composer)

composers_dict = composers.value_counts().to_dict()

df_train['composer_count'] = df_train.composer.apply(lambda x: composers_dict[x])

df_test['composer_count'] = df_train.composer.apply(lambda x: composers_dict[x])



print("[count] lyricist")

lyricists = df_train.lyricist.append(df_test.lyricist)

lyricists_dict = lyricists.value_counts().to_dict()

df_train['lyr_count'] = df_train.lyricist.apply(lambda x: lyricists_dict[x])

df_test['lyr_count'] = df_train.lyricist.apply(lambda x: lyricists_dict[x])



print("[is_radio]")

df_train['is_radio'] = df_train.source_system_tab.apply(lambda x: str(x) == 'radio').astype(np.uint8)



df_test['is_radio'] = df_test.source_system_tab.apply(lambda x: str(x) == 'radio').astype(np.uint8)



print("[is_my_lib]")

df_train['is_my_lib'] = df_train.source_system_tab.apply(lambda x: str(x) == 'my library').astype(np.uint8)

df_test['is_my_lib'] = df_test.source_system_tab.apply(lambda x: str(x) == 'my library').astype(np.uint8)



predictions = np.zeros(shape=[len(df_test)])

idx = 0

TIMES = 0

for train_indices,val_indices in kf.split(df_train) : 

    train_data = lgb.Dataset(df_train.loc[train_indices,['is_radio','is_my_lib','composer_count','source_system_tab','source_screen_name','source_type','msno']],label=df_train.loc[train_indices,'target'])

    val_data = lgb.Dataset(df_train.loc[val_indices,['is_radio', 'is_my_lib', 'composer_count','source_system_tab','source_screen_name','source_type','msno']],label=df_train.loc[val_indices,'target'])

    idx+=1

    if idx != SPLITS:

        continue

#     train_data = lgb.Dataset(df_train.drop(['target'],axis=1),label=df_train['target'])

#     val_data = train_data

    params = {

        'objective': 'binary',

        'metric': 'auc',

        'boosting': 'gbdt',

        'learning_rate': 0.2 ,

        'verbose': 0,

        'num_leaves': 256,

        'bagging_fraction': 1.0,

        'bagging_freq': 1,

        'bagging_seed': 1,

        'feature_fraction': 1.0,

        'feature_fraction_seed': 1,

        'max_bin': 128,

       # 'num_threads':1,

        'early_stopping_round':100,

#          'lambda_l2':0.007,

        } 

#64 leaf --> 0.78716, 1600rounds

#90 leaf --> 0.78842, 1340rounds

#108 leaf --> 0.78881, 940rounds

#150 leaf --> [511]	valid_0's auc: 0.789495

#712 leaf --> highest, 0.67899, 225 rounds

    bst = lgb.train(params, train_data, 3000, valid_sets=[val_data])

    import time

    bst.save_model('model_' + str(time.time()))

    print("predicting..")

    TIMES+=1

    predictions+=bst.predict(df_test.loc[:,['is_radio', 'is_my_lib','composer_count','source_system_tab','source_screen_name','source_type','msno']])



predictions/=TIMES

print("preparing submission..")

submission = pd.read_csv(FOLDER + 'sample_submission.csv')

submission.target=predictions

print("saving submission..")

submission.to_csv('submission_' +str(time.time())+'.csv',index=False)