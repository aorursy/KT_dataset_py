

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')[:50000]

print(train.shape,train.head())

songs = pd.read_csv('../input/songs.csv')

print(songs.head())

#skipped songs get rating 1, listened songs get rating 5

train['rating']=train['target']*5-1

#top songs in log ?



topusers=train.groupby(by=['msno'])['rating'].sum()



topsongs=train.groupby(by=['song_id'])['rating'].sum()

topsongs=topsongs.sort_values(0,ascending=False)   #[:20000]

print(topsongs)

#3.5M songs, we limit to top 30K songs



#it doens't change ranking very much
def trans2vect(df,uid,pid,rate,top):

    from scipy.sparse import csr_matrix

    from sklearn.preprocessing import normalize

    

    #sparse matrix with product in rows and users in columns

    df=df[df['song_id'].isin(top.index)]

    user_u = list(df[uid].unique())

    song_u = list(top.index)

    col = df[uid].astype('category', categories=user_u).cat.codes

    row = df[pid].astype('category', categories=song_u).cat.codes

    songrating = csr_matrix((df[df[pid].isin(song_u)][rate].tolist(), (row,col)), shape=(len(song_u),len(user_u)))

    

    #normalize

    songrating_n = normalize(songrating, norm='l1', axis=0)

    return songrating_n



ratings = trans2vect(train,'msno','song_id','rating',topsongs)

ratings

from scipy.spatial.distance import cosine

from sklearn.metrics.pairwise import cosine_similarity

       

similarities = cosine_similarity(ratings)  #goes south with >15k songs

print(similarities.shape)

similarities
similarusers = cosine_similarity(ratings.T)  #goes south with >15k songs

print(similarusers.shape)

similarusers
similaritiespd = pd.DataFrame(similarities,index=topsongs.index)  #add titles



similar_songs=pd.DataFrame(topsongs.index)

for xi in range(0,10):

    similar_songs[xi]=''



#example song 0



tmp=similaritiespd.loc[:,0:0] #.sort_values(ascending=False)[:10])

print(tmp.sort_values(0,ascending=False)[:10])



for i in range(0,20):

    tmp= similaritiespd.sort_values(i,ascending=False)[:10].index

    for xi in range(0,10):

        similar_songs.iat[i,xi] = tmp[xi]

    

similar_songs
user_u = list(train['msno'].unique())

similarusersspd = pd.DataFrame(similarusers,index=user_u)  #add titles



#example user 0

#

tmp=similarusersspd.loc[:,0:0] #.sort_values(ascending=False)[:10])

print(tmp.sort_values(0,ascending=False)[:10])