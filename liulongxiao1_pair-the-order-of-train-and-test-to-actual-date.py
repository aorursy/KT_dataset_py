import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

pd.options.display.max_rows=1000

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

songs=pd.read_csv('../input/songs.csv')

songs_extra = pd.read_csv('../input/song_extra_info.csv')
def isrc_to_year(isrc):

    if type(isrc) == str:

        if int(isrc[5:7]) > 17:

            return 1900 + int(isrc[5:7])

        else:

            return 2000 + int(isrc[5:7])

    else:

        return np.nan

        

songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)

songs_extra.drop(['isrc'], axis = 1, inplace = True)



train = train.merge(songs_extra, on = 'song_id', how = 'left')

test = test.merge(songs_extra, on = 'song_id', how = 'left')

train = train.merge(songs, on = 'song_id', how = 'left')

test = test.merge(songs, on = 'song_id', how = 'left')

train['id']=train.index

test['id']=test.index

train=train[train['song_year']>=2016]

test=test[test['song_year']>=2016]
train_count=train.groupby('song_id').agg({'song_id':'count'})

test_count=test.groupby('song_id').agg({'song_id':'count'})
train_count=train_count[train_count['song_id']>100]

test_count=test_count[test_count['song_id']>100]
train_detect_set=set(train_count.index)

test_detect_set=set(test_count.index)
train=train[train['song_id'].isin(train_detect_set)]

test=test[test['song_id'].isin(test_detect_set)]
train_first_in=train.groupby('song_id').apply(lambda x:x.id.iloc[0])

test_first_in=test.groupby('song_id').apply(lambda x:x.id.iloc[0])

train_first_in=train_first_in[train_first_in>100000]

test_first_in=test_first_in[test_first_in>100000]

train_first_in.name='first_id'

test_first_in.name='first_id'

train_first_in=train_first_in.reset_index()

test_first_in=test_first_in.reset_index()

train_first_in=train_first_in.merge(songs_extra,on='song_id',how='left')

test_first_in=test_first_in.merge(songs_extra,on='song_id',how='left')
train_first_in=train_first_in.sort_values('first_id')

test_first_in=test_first_in.sort_values('first_id')

train_first_in=train_first_in.merge(songs[['song_id','artist_name','composer']],on='song_id')

test_first_in=test_first_in.merge(songs[['song_id','artist_name','composer']],on='song_id')
train_first_in
test_first_in