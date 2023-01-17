

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv') #[:50000]

print(train.shape,train.head())

songs = pd.read_csv('../input/songs.csv')

print(songs.head())

train['rating']=1

#train the skippers



train=train[train['target']==0]

#top target skipped songs, needs rating to sum

topsongs=train.groupby(by=['song_id'])['rating'].sum()

topsongs=topsongs.sort_values(0,ascending=False)[:10000]

print(topsongs)

#3.5M songs, we limit to top 30K songs



#it doens't change ranking very much
from scipy.sparse import csr_matrix

train=train[train['song_id'].isin(topsongs.index)]



user_u = list(train.msno.unique())

#song_u = list(train.song_id.unique())  #sorted automatically

song_u = list(topsongs.index)



col = train.msno.astype('category', categories=user_u).cat.codes

row = train.song_id.astype('category', categories=song_u).cat.codes



songrating = csr_matrix((train[train['song_id'].isin(song_u)]['rating'].tolist(), (row,col)), shape=(len(song_u),len(user_u)))

#3.5M songs 30k users

songrating
from scipy.spatial.distance import cosine

from sklearn.metrics.pairwise import cosine_similarity

       

similarities = cosine_similarity(songrating)



similiarties=similarities.astype(np.float32)#goes south with >15k songs



print(similarities.shape)

similarities
similaritiespd = pd.DataFrame(similarities,index=song_u)  #add titles



similar_songs=pd.DataFrame(song_u)

for xi in range(0,10):

    similar_songs[xi]=''



#example song 2

tmp=similaritiespd.loc[:,2:2] #.sort_values(ascending=False)[:10])

print(tmp.sort_values(2,ascending=False)[:10])



for i in range(0,100):

    tmp= similaritiespd.sort_values(i,ascending=False)[:10].index

    for xi in range(0,10):

        similar_songs.iat[i,xi] = tmp[xi]

    

similar_songs