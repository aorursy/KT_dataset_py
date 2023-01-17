# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
members_df = pd.read_csv("../input/members.csv")

songs_df = pd.read_csv("../input/songs.csv")

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
print("{:,} users".format(members_df.shape[0]))

members_df.head()
print("{:,} songs".format(songs_df.shape[0]))

songs_df.head()
print("{:,} training records".format(train_df.shape[0]))

train_df.head()
print("{:,} test records".format(test_df.shape[0]))

test_df.head()
#combine the data sources together

train_merged_df = train_df.merge(songs_df, on=["song_id"]).merge(members_df, on=["msno"])
print("target_values = {}".format(list(set(train_df.target.values))))

print("{:.2f}% 0's, {:.2f} 1's".format(

    np.sum(train_df.target == 0)/float(train_df.shape[0])*100.0, 

    np.sum(train_df.target == 1)/float(train_df.shape[0])*100.0))

corr = train_merged_df.corr()

sns.heatmap(corr,  xticklabels=corr.columns.values, yticklabels=corr.columns.values)

plt.title('feature correlations')
plt.figure(figsize=(15,8))

plt.subplot(2,1,1)

bins = np.logspace(2,8, 100)

plt.hist(songs_df.song_length.values, bins=bins, log=False);

plt.xscale('log')

plt.title("song length histogram"); plt.ylabel("count");



plt.subplot(2,1,2)

bins = np.logspace(2,8, 100)

plt.hist(songs_df.song_length.values, bins=bins, log=True);

plt.xscale('log')

plt.title("log song length histogram"); plt.xlabel("song length"); plt.ylabel("count")
plt.figure(figsize=(15,4))

artist_song_counts = list(songs_df["artist_name"].value_counts())

bins = np.logspace(0,5, 100)

plt.hist(artist_song_counts, bins=bins, log=True);

plt.xscale('log')

plt.title("artist song counts"); 

plt.xlabel("# songs"); plt.ylabel("# artists");
plt.figure(figsize=(15,4))

user_song_counts = list(train_df["msno"].value_counts())

plt.figure(figsize=(15,6))



plt.subplot(2,1,1)

bins = np.logspace(0,5, 100)

plt.hist(artist_song_counts, bins=bins, log=True);

plt.xscale('log')

plt.title("user song counts"); 

plt.ylabel("# users");



plt.subplot(2,1,2)

bins = np.logspace(0,5, 100)

plt.hist(artist_song_counts, bins=bins, log=False);

plt.xscale('log')

plt.title("user song counts"); 

plt.xlabel("# songs"); plt.ylabel("# users");
plt.figure(figsize=(15,4))

user_song_counts = list(test_df["msno"].value_counts())

plt.figure(figsize=(15,6))



plt.subplot(2,1,1)

bins = np.logspace(0,5, 100)

plt.hist(artist_song_counts, bins=bins, log=True);

plt.xscale('log')

plt.title("user song counts"); 

plt.ylabel("# users");



plt.subplot(2,1,2)

bins = np.logspace(0,5, 100)

plt.hist(artist_song_counts, bins=bins, log=False);

plt.xscale('log')

plt.title("user song counts"); 

plt.xlabel("# songs"); plt.ylabel("# users");
print("{:,} users in train\n{:,} users in test\n{:,} users in both".format(

    train_df['msno'].values.shape[0], test_df['msno'].values.shape[0],

    len(set(train_df['msno'].values).intersection(set(test_df['msno'].values)))))