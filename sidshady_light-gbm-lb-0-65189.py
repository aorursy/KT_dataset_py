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

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
df_songs = pd.read_csv("../input/songs.csv")
df_songs_extra = pd.read_csv("../input/song_extra_info.csv")



df_members = pd.read_csv("../input/members.csv",parse_dates=["registration_init_time","expiration_date"])
df_test = pd.read_csv("../input/test.csv")
df_train =df_train.merge(df_songs,how="left",on="song_id")
df_train = df_train.merge(df_members,how="left",on="msno")
df_test =df_test.merge(df_songs,how="left",on="song_id")
df_test = df_test.merge(df_members,how="left",on="msno")
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
df_train['language'].fillna(value=df_train['language'].mode()[0],inplace=True)



df_test['language'].fillna(value=df_test['language'].mode()[0],inplace=True)
df_train['genre_ids'] = df_train['genre_ids'].str.split("|")



df_test['genre_ids'] = df_test['genre_ids'].str.split("|")
df_train['genre_count'] = df_train['genre_ids'].apply(lambda x : len(x) if "Unknown" not in x else 0)



df_test['genre_count'] = df_test['genre_ids'].apply(lambda x : len(x) if "Unknown" not in x else 0)
df_train['artist_name'].nunique()
df_test['artist_name'].nunique()
print("Number of Common Artists in both train & Test : ", len(set.intersection(set(df_train['artist_name']), set(df_test['artist_name']))))
df_artists = df_train.loc[:,["artist_name","target"]]



artists1 = df_artists.groupby(["artist_name"],as_index=False).sum().rename(columns={"target":"repeat_count"})



artists2 = df_artists.groupby(["artist_name"],as_index=False).count().rename(columns = {"target":"play_count"})
df_artist_repeats = artists1.merge(artists2,how="inner",on="artist_name")
df_artist_repeats.head()
df_artist_repeats["repeat_percentage"] = round((df_artist_repeats['repeat_count']*100) / df_artist_repeats['play_count'],1)
df_artist_repeats.head()
df_artist_repeats.drop(["repeat_count","play_count"],axis=1,inplace=True)
df_train = df_train.merge(df_artist_repeats,on="artist_name",how="left").rename(columns={"repeat_percentage":"artist_repeat_percentage"})
df_test = df_test.merge(df_artist_repeats,on="artist_name",how="left").rename(columns={"repeat_percentage":"artist_repeat_percentage"})
df_train.drop(["genre_ids","artist_name"],axis=1,inplace=True)



df_test.drop(["genre_ids","artist_name"],axis=1,inplace=True)
del df_artist_repeats

del df_artists
df_train['composer'] = df_train['composer'].str.split("|") 



df_test['composer'] = df_test['composer'].str.split("|") 
df_train['composer_count'] = df_train['composer'].apply(lambda x : len(x) if "Unknown" not in x else 0 )



df_test['composer_count'] = df_test['composer'].apply(lambda x : len(x) if "Unknown" not in x else 0 )
df_train['source_system_tab'].value_counts()
source_tab_dict = {"my library":8,"discover":7,"search":6,"radio":5,"listen with":4,"explore":3,"notification":2,"settings":1,"Unknown":0 }



source_screen_name_dict = {"Local playlist more":19,"Online playlist more":18,"Radio":17,"Unknown":16,"Album more":15,"Search":14,"Artist more":13,"Discover Feature":12,"Discover Chart":11,"Others profile more":10,"Discover Genre":9,"My library":8,"Explore":7,"Discover New":6,"Search Trends":5,"Search Home":4,"My library_Search":3,"Self profile more":2,"Concert":1,"Payment":0}



source_type_dict = {"local-library":12,"online-playlist":11,"local-playlist":10,"radio":9,"album":8,"top-hits-for-artist":7,"song":6,"song-based-playlist":5,"listen-with":4,"Unknown":3,"topic-article-playlist":2,"artist":1,"my-daily-playlist":0}

df_train['source_system_tab'] = df_train['source_system_tab'].map(source_tab_dict)



df_test['source_system_tab'] = df_test['source_system_tab'].map(source_tab_dict)
df_train['source_type'] = df_train['source_type'].map(source_type_dict)

df_test['source_type'] = df_test['source_type'].map(source_type_dict)
df_train['source_screen_name'] = df_train['source_screen_name'].map(source_screen_name_dict)

df_test['source_screen_name'] = df_test['source_screen_name'].map(source_screen_name_dict)
gender_train = pd.get_dummies(df_train['gender'],drop_first=True)



gender_test = pd.get_dummies(df_test['gender'],drop_first=True)
df_train = pd.concat([df_train,gender_train],axis=1)

df_test = pd.concat([df_test,gender_test],axis=1)
df_train.drop(["composer","gender"],axis=1,inplace=True)

df_test.drop(["composer","gender"],axis=1,inplace=True)
df_train['validity_days'] = (df_train['expiration_date'] - df_train['registration_init_time']).dt.days



df_test['validity_days'] = (df_test['expiration_date'] - df_test['registration_init_time']).dt.days
df_train.drop(["registration_init_time","expiration_date"],axis=1,inplace=True)



df_test.drop(["registration_init_time","expiration_date"],axis=1,inplace=True)
df_train['lyricist'] = df_train['lyricist'].str.split("|")

df_test['lyricist'] = df_test['lyricist'].str.split("|")
df_train['lyricist_count'] = df_train['lyricist'].apply(lambda x : len(x) if "Unknown" not in x else 0 )



df_test['lyricist_count'] = df_test['lyricist'].apply(lambda x : len(x) if "Unknown" not in x else 0 )
df_test['artist_repeat_percentage'].fillna(value=0.0,inplace=True)
df_test['source_screen_name'].fillna(df_test['source_screen_name'].mode()[0],inplace=True)
df_train.drop("lyricist",axis=1,inplace=True)



df_test.drop("lyricist",axis=1,inplace=True)
df_songs_extra.drop("name",axis=1,inplace=True)
df_train = df_train.merge(df_songs_extra,how="left",on="song_id")
df_test = df_test.merge(df_songs_extra,how="left",on="song_id")
def isrc_to_year(isrc):

    if type(isrc) == str:

        if int(isrc[5:7]) > 17:

            return 1900 + int(isrc[5:7])

        else:

            return 2000 + int(isrc[5:7])

    else:

        return np.nan
df_train['song_year'] = df_train['isrc'].apply(isrc_to_year)



df_test['song_year'] = df_test['isrc'].apply(isrc_to_year)
df_train.drop("isrc",axis=1,inplace=True)

df_test.drop("isrc",axis=1,inplace=True)
df_train['song_year'].fillna(value=-1,inplace=True)



df_test['song_year'].fillna(value=-1,inplace=True)
df_train['song_year'] = df_train['song_year'].astype("int")



df_test['song_year'] = df_test['song_year'].astype("int")
df_train['source_system_tab'] = df_train['source_system_tab'].astype("category")

df_test['source_system_tab'] = df_test['source_system_tab'].astype("category")



df_train['source_screen_name'] = df_train['source_screen_name'].astype("category")

df_test['source_screen_name'] = df_test['source_screen_name'].astype("category")



df_train['source_type'] = df_train['source_type'].astype("category")

df_test['source_type'] = df_test['source_type'].astype("category")



df_train['language'] = df_train['language'].astype("category")

df_test['language'] = df_test['language'].astype("category")



df_train['city'] = df_train['city'].astype("category")

df_test['city'] = df_test['city'].astype("category")





df_train['registered_via'] = df_train['registered_via'].astype("category")

df_test['registered_via'] = df_test['registered_via'].astype("category")

df_train['age_range'] = pd.cut(df_train['bd'],bins=[-45,0,10,18,35,50,80,1052])

df_test['age_range'] = pd.cut(df_test['bd'],bins=[-45,0,10,18,35,50,80,1052])



combine = [df_train, df_test]



for dataset in combine : 

    

    dataset.loc[(dataset['bd'] > 0) & (dataset['bd'] <= 10), 'age_category'] = 0

    dataset.loc[(dataset['bd'] > 80) & (dataset['bd'] <= 1052), 'age_category'] = 1

    dataset.loc[(dataset['bd'] > 50) & (dataset['bd'] <= 80), 'age_category'] = 2

    dataset.loc[(dataset['bd'] > 10) & (dataset['bd'] <= 18), 'age_category'] = 3

    dataset.loc[(dataset['bd'] > 35) & (dataset['bd'] <= 50), 'age_category'] = 4

    dataset.loc[(dataset['bd'] > -45) & (dataset['bd'] <= 0), 'age_category'] = 5

    dataset.loc[(dataset['bd'] > 18) & (dataset['bd'] <= 35), 'age_category'] = 6
df_train.drop(["age_range","bd"],axis=1,inplace=True)

df_test.drop(["age_range","bd"],axis=1,inplace=True)
X = df_train.drop(["msno","song_id","target"],axis=1).values



y = df_train['target'].values
import lightgbm as lgb



d_train = lgb.Dataset(X, y)

watchlist = [d_train]
params = {}

params['learning_rate'] = 0.5

params['application'] = 'binary'

params['max_depth'] = 10

params['num_leaves'] = 2**6

params['verbosity'] = 0

params['metric'] = 'auc'
model = lgb.train(params, train_set=d_train, num_boost_round=60, valid_sets=watchlist, \

verbose_eval=5)
song_ids = df_test['id'].values



X_test = df_test.drop(["msno","song_id","id"],axis=1).values
y_preds = model.predict(X_test)
result_df = pd.DataFrame()



result_df['id'] = song_ids

result_df['target'] = y_preds
result_df.to_csv('submission_new.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')