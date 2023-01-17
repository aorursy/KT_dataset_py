import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from itertools import chain
print('Loading data...')

data_path = '../input/'

train = pd.read_csv(data_path + 'train.csv')

test = pd.read_csv(data_path + 'test.csv')

members = pd.read_csv(data_path + 'members.csv', 

                      parse_dates=["registration_init_time", "expiration_date"])

members = members[members["expiration_date"] != "1970-01-01"].copy()

songs = pd.read_csv(data_path + 'songs.csv')
members_train = set(train["msno"].values)

members_test = set(test["msno"].values)

print("# of Members in train:\t", len(members_train))

print("# of Members in test:\t", len(members_test))
print("# of train only:\t", len(members_train - members_test))

print("# of test only: \t", len(members_test - members_train))

print("# of both:      \t", len(members_test & members_train))
members["scope"] = "both"

members.loc[members.msno.isin(members_train - members_test), "scope"] = "train"

members.loc[members.msno.isin(members_test - members_train), "scope"] = "test"
songs_train = set(train['song_id'].values)

songs_test = set(test['song_id'].values)

print("# of Songs in train:\t", len(songs_train))

print("# of Songs in test:\t", len(songs_test))
print("# of train only:\t", len(songs_train - songs_test))

print("# of test only: \t", len(songs_test - songs_train))

print("# of both:      \t", len(songs_test & songs_train))
genre_ids_train = train[['song_id']].merge(songs[['song_id', 'genre_ids']], on='song_id', how='left')

genre_ids_test = test[['song_id']].merge(songs[['song_id', 'genre_ids']], on='song_id', how='left')



genre_ids_train_set = set(chain.from_iterable(

    map(lambda x: str(x).split('|'), genre_ids_train['genre_ids'].unique())))



genre_ids_test_set = set(chain.from_iterable(

    map(lambda x: str(x).split('|'), genre_ids_test['genre_ids'].unique())))



print("# of Genre ids in train:\t", len(genre_ids_train_set))

print("# of Genre ids in test: \t", len(genre_ids_test_set))
print("# of train only:\t", len(genre_ids_train_set - genre_ids_test_set))

print("# of test only: \t", len(genre_ids_test_set - genre_ids_train_set))

print("# of both:      \t", len(genre_ids_train_set & genre_ids_test_set))