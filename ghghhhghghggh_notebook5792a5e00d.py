# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
u_cols = [
'user_id', 'age', 'sex',
'occupation', 'zip_code'
]
users = pd.read_csv('../input/ml100k/u.user',
sep='|',
names = u_cols, encoding='latin-1')
# Load movie data
r_cols = [
'user_id', 'movie_id',
'rating', 'unix_timestamp'
]
ratings = pd.read_csv('../input/ml100k/u.data',
sep='\t',
names=r_cols, encoding='latin-1')
# The movie file contains columns indicating the genres of the movie
# We will only load the first three columns of the file with
m_cols = [
'movie_id', 'title',
'release_date'
]
movies = pd.read_csv('../input/ml100k/u.item',
sep='|',
names=m_cols ,
usecols= range (3), encoding='latin-1')
# Create a DataFrame using only the fields required
data = pd.merge(pd. merge(ratings , users), movies)
data = data[[ 'user_id', 'title', 'movie_id', 'rating']]
print ("The BD has " + str(data.shape[0]) +" ratings")
print ("The BD has ", data.user_id.nunique()," users")
print ("The BD has ", data. movie_id.nunique(), " items")
print (data.head())
data_train = data[0:9000]
# dataframe with the data from user 1
df_usr1 = data_train[data_train.user_id == 1]
# dataframe with the data from user 2
df_usr2 = data_train[data_train.user_id == 6]
# We first compute the set of common movies
common_mov = set(df_usr1.movie_id ).intersection(
df_usr2.movie_id)
print ("\nNumber of common movies",
len(common_mov))
# Sub -dataframe with only the common movies
mask = ( df_usr1.movie_id. isin( common_mov))
data_user_1 = df_usr1[ mask]
print (df_usr1[[ 'title', 'rating']].head())
mask = ( df_usr2.movie_id. isin( common_mov))
data_user_2 = df_usr2[ mask]
print (df_usr2[[ 'title', 'rating']].head())

from scipy.spatial.distance import euclidean
# Similarity based on Euclidean distance for users 1-2
def SimEuclid(df ,User1 ,User2 , min_common_items =10):
# GET MOVIES OF USER1
    mov_u1 = df[df['user_id'] == User1 ]
# GET MOVIES OF USER2
    mov_u2 = df[df['user_id'] == User2 ]
# FIND SHARED FILMS
    rep = pd.merge( mov_u1 , mov_u2 , on = 'movie_id')
    if len(rep) == 0:
        return 0
    if(len(rep) < min_common_items):
        return 0
    return 1.0 / (1.0+ euclidean(rep['rating_x'] ,rep['rating_y']))
from scipy.stats import pearsonr
# Similarity based on Pearson correlation for user 1-2
def SimPearson(df , User1 , User2 , min_common_items = 10):
    # GET MOVIES OF USER1
    mov_u1 = df[df['user_id'] == User1 ]
    # GET MOVIES OF USER2
    mov_u2 = df[df['user_id'] == User2 ]
    # FIND SHARED FILMS
    rep = pd.merge( mov_u1 , mov_u2 , on = 'movie_id')
    if len(rep)==0:
        return 0
    if(len(rep) < min_common_items):
        return 0
    return pearsonr(rep['rating_x'], rep['rating_y']) [0]
print ("Euclidean similarity",float(SimEuclid( data_train , 1, 8)))
print ("Pearson similarity",float(SimPearson( data_train , 1, 8)))
print ("Euclidean similarity",float(SimEuclid( data_train , 1, 31)))
print ("Pearson similarity",float(SimPearson( data_train , 1, 31)))
def assign_to_set(df):
    sampled_ids = np. random.choice(
        df.index ,
        size = np.int64(np.ceil(df.index.size * 0.2)),
        replace=False)
    df.loc[sampled_ids , 'for_testing'] = True
    return df
data['for_testing'] = False
grouped = data.groupby('user_id', group_keys = False).apply (assign_to_set)
X_train = data[grouped.for_testing == False]
X_test = data[grouped.for_testing == True]
def compute_rmse(y_pred , y_true):
    return np.sqrt(np.mean (np.power(y_pred - y_true , 2)))
class CollaborativeFiltering:
    def __init__(self , df , similarity = SimPearson):
        self.sim_method = similarity
        self.df = df
        self.sim = pd.DataFrame(
        np.sum ([0]), columns = df.user_id.unique(),
        index = df.user_id.unique())
    def fit(self):
        
        allUsers = set(self.df['user_id'])
        self.sim = {}
        for person1 in allUsers:
            self.sim.setdefault( person1 , {})
            a = self.df[self.df['user_id'] == person1][[ 'movie_id']]
            data_reduced = pd. merge(self.df , a, on = 'movie_id')
        for person2 in allUsers:
            # Avoid our -self
            if person1 == person2: 
                continue
            self.sim.setdefault( person2 , {})
            if( person1 in self.sim[person2]):
                continue 
            sim = self.sim_method( data_reduced ,person1,person2)
            if(sim < 0):
                self.sim[person1][person2] = 0
                self.sim[person2][person1] = 0
            else :
                self.sim[person1][person2] = sim
                self.sim[person2][person1] = sim
    def predict(self , user_id , movie_id):
        totals = {}
        users = self.df[self.df['movie_id'] == movie_id]
        rating_num , rating_den = 0.0, 0.0
        allUsers = set(users['user_id'])
        for other in allUsers:
            if user_id == other: continue
            rating_num += self.sim[user_id][other] * float (users[users['user_id'] == other][ 'rating'])
            rating_den += self.sim[user_id][other]
            if rating_den == 0:
                if self.df.rating [self.df['movie_id'] == movie_id]. mean() > 0:
                    return self.df.rating[self.df['movie_id'] == movie_id]. mean()
                else :
                    return self.df.rating[self.df['user_id'] == user_id]. mean()
            return rating_num/rating_den
    def evaluate(fit_f ,train , test):
        ids_to_estimate = zip(test.user_id , test. movie_id)
        estimated = np. array([fit_f(u, i)
            if u in train.user_id
            else 3
            for (u, i)
            in ids_to_estimate])
        real = test.rating.values
        return compute_rmse( estimated , real)

def SimPearsonCorrected (df , User1 , User2 ,min_common_items = 1,pref_common_items = 20):
    # GET MOVIES OF USER1
    m_user1 = df[df['user_id'] == User1 ]
    # GET MOVIES OF USER2
    m_user2 = df[df['user_id'] == User2 ]
    # FIND SHARED FILMS
    rep = pd.merge( m_user1 , m_user2 , on = 'movie_id')
    if len(rep) == 0: return 0
    if(len(rep) < min_common_items): return 0
    res = pearsonr(rep['rating_x'], rep['rating_y'])[0]
    res = res * min(pref_common_items , len(rep))
    res = res / pref_common_items
    if(np.isnan(res)): return 0
    return res
reco4 = CollaborativeFiltering(data_train , similarity = SimPearsonCorrected)
reco4.fit()
print ('RMSE for Collaborative Recommender:')
print ('%s' % evaluate( reco4.fit , data_train , data_test))
