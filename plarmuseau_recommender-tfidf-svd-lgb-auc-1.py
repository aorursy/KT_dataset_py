

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')[:100000]



songs = pd.read_csv('../input/songs.csv')

print(songs.head())

members = pd.read_csv('../input/members.csv')

#print(members.head(20))

train=train.merge(members, left_on='msno', right_on='msno', how='inner')

train['teller']=1.0

print(train.shape,train.head())
#skipped songs get rating 1, listened songs get rating 5

train['rating']=train['target']*5-1

train.fillna(value=0,axis=1,inplace=True)



topusers=train.groupby(by=['msno'])['rating'].sum()

topsongs=train.groupby(by=['song_id'])['rating'].sum()

topsongs=topsongs.sort_values(0,ascending=False)   #[:20000]

print(topsongs)

#3.5M songs, we limit to top 30K songs

def trans2vect(df,uid,pid,rate):

    from scipy.sparse import csr_matrix

    from sklearn.preprocessing import normalize

    toppid=train.groupby(by=[pid])[rate].sum()

    toppid=toppid.sort_values(0,ascending=False)   #[:20000]

    print(toppid)    

    #sparse matrix with product in rows and users in columns

    df=df[df[pid].isin(toppid.index)]

    user_u = list(df[uid].unique())

    song_u = list(toppid.index)

    col = df[uid].astype('category', categories=user_u).cat.codes

    row = df[pid].astype('category', categories=song_u).cat.codes

    songrating = csr_matrix((df[df[pid].isin(song_u)][rate].tolist(), (row,col)), shape=(len(song_u),len(user_u)))

    

    #normalize

    songrating_n = normalize(songrating, norm='l1', axis=0)

    return songrating_n,toppid.index



ratings,songU = trans2vect(train,'msno','song_id','teller')

ratings



gender,genderU =trans2vect(train,'msno','gender','teller')

gender

from scipy.sparse import vstack

ratings=vstack((ratings,gender))

ratings
bds,bdU =trans2vect(train,'msno','bd','teller')

bds

from scipy.sparse import vstack

ratings=vstack((ratings,bds))

ratings
registered,registerU =trans2vect(train,'msno','registered_via','teller')

registered

from scipy.sparse import vstack

ratings=vstack((ratings,registered))

ratings
from scipy.spatial.distance import cosine

from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse.linalg import svds       

#similarities = cosine_similarity(ratings)  #goes south with >15k songs

#print(similarities.shape)

#similarities



SongsU,Eigen,UsersU=svds(ratings, k=100)



#print(Eigen)

#print(SongsU)
userU = list(train['msno'].unique())

index1=songU.append(genderU)

index2=index1.append(bdU)

index3=index2.append(registerU)

similarities = pd.DataFrame( cosine_similarity(SongsU,UsersU.T) , index=index3,columns=userU)



#find similar users

similarities.sort_values('T86YHdD4C9JSc274b1IlMkLuNdz4BQRB50fWWE7hx9g=',axis=1)

#find top songs to recommend

similarities.sort_values('FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=',axis=0)[-10:]
#find the third train record

similarities.loc[train.iloc[3]['song_id'],train.iloc[3]['msno']]
train['sim']=0.0

print(train.columns)



for xi in range(0,len(train)):

    train.iat[xi,14]=similarities.loc[train.iloc[xi]['song_id'],train.iloc[xi]['msno']]

    
print( train.groupby(by=['target'])['sim'].mean() )

print( train.groupby(by=['target'])['sim'].std() )

#split between 0 and 0.07
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from tqdm import tqdm



test = pd.read_csv('../input/test.csv')[:50000]

test=test.merge(members, left_on='msno', right_on='msno', how='inner')



train = train.fillna(-1)

test = test.fillna(-1)



cols = list(train.columns)

cols.remove('target')



for col in tqdm(cols):

    if train[col].dtype == 'object':

        train[col] = train[col].apply(str)

        test[col] = test[col].apply(str)



        le = LabelEncoder()

        train_vals = list(train[col].unique())

        test_vals = list(test[col].unique())

        le.fit(train_vals + test_vals)

        train[col] = le.transform(train[col])

        test[col] = le.transform(test[col])



        print(col + ': ' + str(len(train_vals)) + ', ' + str(len(test_vals)))



X = np.array(train.drop(['target','rating'], axis=1))

y = train['target'].values



X_test = np.array(test.drop(['id'], axis=1))

ids = test['id'].values



del train, test



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state = 12)



d_train = lgb.Dataset(X_train, label=y_train)

d_valid = lgb.Dataset(X_valid, label=y_valid) 



watchlist = [d_train, d_valid]





print('Training LGBM model...')

params = {}

params['learning_rate'] = 0.4

params['application'] = 'binary'

params['max_depth'] = 15

params['num_leaves'] = 2**8

params['verbosity'] = 0

params['metric'] = 'auc'



model = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, \

early_stopping_rounds=50, verbose_eval=10)
