# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt 

%matplotlib inline

from subprocess import check_output

import time





print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
start_time = time.time()

print("Merge Start....")

df_train = pd.read_csv("../input/train.csv")

df_songs = pd.read_csv("../input/songs.csv")



df_songs_extra = pd.read_csv("../input/song_extra_info.csv")



df_members = pd.read_csv("../input/members.csv",parse_dates=["registration_init_time","expiration_date"])

df_test = pd.read_csv("../input/test.csv")

df_train =df_train.merge(df_songs,how="left",on="song_id")

df_train = df_train.merge(df_members,how="left",on="msno")

df_test =df_test.merge(df_songs,how="left",on="song_id")

df_test = df_test.merge(df_members,how="left",on="msno")

print("Merge End")

end_time = time.time()



print(end_time-start_time)
df_train.head()

df_train.count()


df_train['gender'].fillna(value="Unknown",inplace=True)

df_test['gender'].fillna(value="Unknown",inplace=True)



df_train['source_system_tab'].fillna(value="Unknown",inplace=True)

df_test['source_system_tab'].fillna(value="Unknown",inplace=True)



df_train['source_screen_name'].fillna(value="Unknown",inplace=True)

df_test['source_screen_name'].fillna(value="Unknown",inplace=True)



df_train['source_type'].fillna(value="Unknown",inplace=True)

df_test['source_type'].fillna(value="Unknown",inplace=True)



df_train['genre_ids'].fillna(value="Unknown",inplace=True)

df_test['genre_ids'].fillna(value="Unknown",inplace=True)



df_train['artist_name'].fillna(value="Unknown",inplace=True)

df_test['artist_name'].fillna(value="Unknown",inplace=True)



df_train['composer'].fillna(value="Unknown",inplace=True)

df_test['composer'].fillna(value="Unknown",inplace=True)



df_train['lyricist'].fillna(value="Unknown",inplace=True)

df_test['lyricist'].fillna(value="Unknown",inplace=True)



df_train['song_length'].fillna(value=df_train['song_length'].mean(),inplace=True)

df_test['song_length'].fillna(value=df_test['song_length'].mean(),inplace=True)

#song_length null -> mean



df_train['language'].fillna(value=df_train['language'].mode()[0],inplace=True)

df_test['language'].fillna(value=df_test['language'].mode()[0],inplace=True)

#langeuage -> mode
df_train['genre_ids'] = df_train['genre_ids'].str.split("|")

df_test['genre_ids'] = df_test['genre_ids'].str.split("|")
df_train['genre_count'] = df_train['genre_ids'].apply(lambda x : len(x) if "Unknown" not in x else 0)

df_test['genre_count'] = df_test['genre_ids'].apply(lambda x : len(x) if "Unknown" not in x else 0)
def isrc_to_year(isrc):

    if type(isrc) == str:

        if int(isrc[5:7]) > 17:

            return int(isrc[5:7])//10

        else:

            return int(isrc[5:7])//10

    else:

        return np.nan

        

df_songs_extra['song_year'] = df_songs_extra['isrc'].apply(isrc_to_year)

df_songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)



df_train = df_train.merge(df_songs_extra, on = 'song_id', how = 'left')

df_test = df_test.merge(df_songs_extra, on = 'song_id', how = 'left')



df_train['song_year'].fillna(value="Unknown",inplace=True)

df_test['song_year'].fillna(value="Unknown",inplace=True)



#df_train['song_year'] = pd.to_numeric(df_train['song_year'],downcast='integer')

#df_test['song_year'] = pd.to_numeric(df_test['song_year'],downcast='integer')



source_tab_dict = {"my library":8,"discover":7,"search":6,"radio":5,"listen with":4,"explore":3,"notification":2,"settings":1,"Unknown":0 ,"null":9}

source_screen_name_dict = {"Local playlist more":19,"Online playlist more":18,"Radio":17,"Unknown":16,"Album more":15,"Search":14,"Artist more":13,"Discover Feature":12,"Discover Chart":11,"Others profile more":10,"Discover Genre":9,"My library":8,"Explore":7,"Discover New":6,"Search Trends":5,"Search Home":4,"My library_Search":3,"Self profile more":2,"Concert":1,"Payment":0}

source_type_dict = {"local-library":12,"online-playlist":11,"local-playlist":10,"radio":9,"album":8,"top-hits-for-artist":7,"song":6,"song-based-playlist":5,"listen-with":4,"Unknown":3,"topic-article-playlist":2,"artist":1,"my-daily-playlist":0}
df_train['source_system_tab'] = df_train['source_system_tab'].map(source_tab_dict)

df_test['source_system_tab'] = df_test['source_system_tab'].map(source_tab_dict)



df_train['source_type'] = df_train['source_type'].map(source_type_dict)

df_test['source_type'] = df_test['source_type'].map(source_type_dict)



df_train['source_screen_name'] = df_train['source_screen_name'].map(source_screen_name_dict)

df_test['source_screen_name'] = df_test['source_screen_name'].map(source_screen_name_dict)



# source_type, source_screen_name mapping
#df_train['language'] = pd.to_numeric(df_train['language'],downcast='integer')
gender_train = pd.get_dummies(df_train['gender'],drop_first=True)

gender_test = pd.get_dummies(df_test['gender'],drop_first=True)



df_train = pd.concat([df_train,gender_train],axis=1)

df_test = pd.concat([df_test,gender_test],axis=1)
# Convert date to number of days

df_train['membership_days'] = (df_train['expiration_date'] - df_train['registration_init_time']).dt.days.astype(int)



# Remove both date fieldsa since we already have the number of days between them

df_train = df_train.drop(['registration_init_time','expiration_date'], axis=1)



# Convert date to number of days

df_test['membership_days'] = (df_test['expiration_date'] - df_test['registration_init_time']).dt.days.astype(int)



# Remove both date fieldsa since we already have the number of days between them

df_test = df_test.drop(['registration_init_time','expiration_date'], axis=1)
df_train.info()
composer = df_train['composer']

artist_name = df_train['artist_name']

city = df_train['city']

gender = df_train['gender']

lyricist = df_train['lyricist']



del df_train['composer']

del df_train['artist_name']

del df_train['city']

del df_train['gender']

del df_train['lyricist']
composer_ = df_test['composer']

artist_name_ = df_test['artist_name']

city_ = df_test['city']

gender_ = df_test['gender']

lyricist_ = df_test['lyricist']







del df_test['composer']

del df_test['artist_name']

del df_test['city']

del df_test['gender']

del df_test['lyricist']
df_train.info()
del df_train['genre_ids']

del df_train['bd']

del df_test['genre_ids']

del df_test['bd']


df_train['song_year'] = df_train['song_year'].astype("category")



df_test['song_year'] = df_test['song_year'].astype("category")
df_train['language']
df_train['female'] = df_train['female'].astype("category")

df_test['female'] = df_test['female'].astype("category")



df_train['male'] = df_train['male'].astype("category")

df_test['male'] = df_test['male'].astype("category")



df_train['genre_count'] = df_train['genre_count'].astype("category")

df_test['genre_count'] = df_test['genre_count'].astype("category")
df_train['source_system_tab'] = df_train['source_system_tab'].astype("category")

df_test['source_system_tab'] = df_test['source_system_tab'].astype("category")



df_train['source_screen_name'] = df_train['source_screen_name'].astype("category")

df_test['source_screen_name'] = df_test['source_screen_name'].astype("category")



df_train['source_type'] = df_train['source_type'].astype("category")

df_test['source_type'] = df_test['source_type'].astype("category")



df_train['language'] = df_train['language'].astype("category")

df_test['language'] = df_test['language'].astype("category")



df_train['registered_via'] = df_train['registered_via'].astype("category")

df_test['registered_via'] = df_test['registered_via'].astype("category")







#for col in df_train.columns:

#    if col != "target":

#        df_train[col] = df_train[col].astype('category')

#        df_test[col] = df_test[col].astype('category')

del df_train['song_year']
df_train.info()

df_train['genre_count'].isnull
del df_train['membership_days']
del df_train['song_length']
X = df_train.drop(["msno","song_id","target"],axis=1).values



y = df_train['target'].values
import lightgbm as lgb



d_train = lgb.Dataset(X, y)

watchlist = [df_train]
params = {}

params['learning_rate'] = 0.5

params['application'] = 'binary'

params['max_depth'] = 10

params['num_leaves'] = 2**6

params['verbosity'] = 0

params['metric'] = 'auc'
model = lgb.train(params, train_set=d_train, num_boost_round=60, valid_sets=watchlist, \

verbose_eval=5)
df_train.info()
del df_train['male']

del df_train['female']
del df_train['language']
del df_train['bd']
X
del df_train['song_year']
df_train.info()
df_train.info()