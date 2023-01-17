import pandas as pd
path = "../input/"
train = pd.read_csv(path+"train.csv")

print(train.shape)

train.head()
members = pd.read_csv(path+"members.csv")

members.registration_init_time = pd.to_datetime(members.registration_init_time, format="%Y%m%d")

members.expiration_date = pd.to_datetime(members.expiration_date, format="%Y%m%d")

## Add a little feature

members["membership_time_left"] = (members.expiration_date - members.registration_init_time).dt.days

print(members.shape)

members.head()
songs = pd.read_csv(path+"songs.csv")

print(songs.shape)

songs.head()
test = pd.read_csv(path+"test.csv")

print(test.shape)

test.head()
train = train.merge(members,on="msno")

train = train.merge(songs,on="song_id")
test = test.merge(members,on="msno")

test = test.merge(songs,on="song_id")
train.head()
# MSNO and song id are useless after this point, but we may want ot keep them in for merging with external data from other kernels, 

## Or for e.g. using the kkbox financial transactions/churn data (Assuming the IDs match). 



# train.drop(["msno","song_id"],axis=1).to_csv("merged_train-kkbox_music.csv.gz",index=False,compression="gzip")
train.to_csv("merged_train-kkbox_music_V1.csv.gz",index=False,compression="gzip")

test.to_csv("merged_test-kkbox_music_V1.csv.gz",index=False,compression="gzip")