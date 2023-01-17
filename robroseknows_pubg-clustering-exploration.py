import gc
import time
# Data
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import random
random.seed(1337)
np.random.seed(1337)

# Credit for this method here: https://www.kaggle.com/rejasupotaro/effective-feature-engineering
def reload():
    gc.collect()
    df = pd.read_csv('../input/train_V2.csv')
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    return df
df = reload()
df.head()
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df = reduce_mem_usage(df)
df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
df = reduce_mem_usage(df)
df.head()
# Dropping columns with too many categories or unique values as well as the target column.
target = 'winPlacePerc'
drop_cols = ['Id', 'groupId', 'matchId', target]
select = [x for x in df.columns if x not in drop_cols]
X = df.loc[:, select]
X.head()
# Now one-hot encode the remaining category column (matchType)
X = pd.get_dummies(X)
X.head()
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
pca2.fit(X)
print(sum(pca2.explained_variance_ratio_))
P2 = pca2.transform(X)
plt.scatter(P2[:100000, 0], P2[:100000, 1])
plt.show()
pca3 = PCA(n_components=3)
pca3.fit(X)
print(sum(pca3.explained_variance_ratio_))
P3 = pca3.transform(X)
from mpl_toolkits.mplot3d import Axes3D
fig_p3 = plt.figure()
ax = Axes3D(fig_p3, elev=48, azim=134)
ax.scatter(P3[:100000, 0], P3[:100000, 1], P3[:100000, 2])
fig_p3.show()
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=2).fit(P2)
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms.labels_[:100000])
plt.show()
kms3 = KMeans(n_clusters=3).fit(P2)
kms4 = KMeans(n_clusters=4).fit(P2)
kms5 = KMeans(n_clusters=5).fit(P2)
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms3.labels_[:100000])
plt.show()
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms4.labels_[:100000])
plt.show()
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms5.labels_[:100000])
plt.show()
kms6 = KMeans(n_clusters=6).fit(P2)
kms7 = KMeans(n_clusters=7).fit(P2)
kms8 = KMeans(n_clusters=8).fit(P2)
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms6.labels_[:100000])
plt.show()
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms7.labels_[:100000])
plt.show()
plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms8.labels_[:100000])
plt.show()
def cluster_features(df, model, pca):
    P = pca.transform(df)
    new_df = pd.DataFrame()
    new_df['cluster'] = model.predict(P)
    one_hot = pd.get_dummies(new_df['cluster'], prefix='cluster')
    new_df = new_df.join(one_hot)
    new_df = new_df.drop('cluster', axis=1)
    new_df = new_df.fillna(0)
    return new_df
    
def centroid_features(df, model, pca):
    P = pd.DataFrame(pca.transform(df))
    new_df = pd.DataFrame()
    cluster = 0
    for centers in model.cluster_centers_:
        new_df['distance_{}'.format(cluster)] = np.linalg.norm(P[[0, 1]].sub(np.array(centers)), axis=1)
        cluster += 1
    return new_df
def norm_features(df):
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
    df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
    df = reduce_mem_usage(df)
    return df

def one_hot_encode(df):
    return pd.get_dummies(df, columns=['matchType'])

def remove_categories(df):
    target = 'winPlacePerc'
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', target]
    select = [x for x in df.columns if x not in drop_cols]
    return df.loc[:, select]
def kmeans_5_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms5, pca2))
    
def kmeans_5_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms5, pca2))

def kmeans_3_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms3, pca2))
    
def kmeans_3_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms3, pca2))

def kmeans_4_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms4, pca2))
    
def kmeans_4_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms4, pca2))
def train_test_split(df, test_size=0.1):
    match_ids = df['matchId'].unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids, train_size)

    train = df[df['matchId'].isin(train_match_ids)]
    test = df[-df['matchId'].isin(train_match_ids)]
    
    return train, test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def run_experiment(preprocess):
    df = reload()    

    df = preprocess(df)
    df.fillna(0, inplace=True)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    train, val = train_test_split(df, 0.1)
    
    model = LinearRegression()
    model.fit(train[cols_to_fit], train[target])
    
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_experiments(preprocesses):
    results = []
    for preprocess in preprocesses:
        start = time.time()
        score = run_experiment(preprocess)
        execution_time = time.time() - start
        results.append({
            'name': preprocess.__name__,
            'score': score,
            'execution time': f'{round(execution_time, 2)}s'
        })
        gc.collect()
        
    return pd.DataFrame(results, columns=['name', 'score', 'execution time']).sort_values(by='score')
def original(df):
    return df

def items(df):
    df['items'] = df['heals'] + df['boosts']
    return df

def players_in_team(df):
    agg = df.groupby(['groupId']).size().to_frame('players_in_team')
    return df.merge(agg, how='left', on=['groupId'])

def total_distance(df):
    df['total_distance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']
    return df

def headshotKills_over_kills(df):
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['headshotKills_over_kills'].fillna(0, inplace=True)
    return df

def killPlace_over_maxPlace(df):
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['killPlace_over_maxPlace'].fillna(0, inplace=True)
    df['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)
    return df

def walkDistance_over_heals(df):
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_heals'].fillna(0, inplace=True)
    df['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)
    return df

def walkDistance_over_kills(df):
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['walkDistance_over_kills'].fillna(0, inplace=True)
    df['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)
    return df

def teamwork(df):
    df['teamwork'] = df['assists'] + df['revives']
    return df

def min_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId','groupId'])[features].min()
    return df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

def max_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])

def sum_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].sum()
    return df.merge(agg, suffixes=['', '_sum'], how='left', on=['matchId', 'groupId'])

def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

def mean_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])

def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
run_experiments([
    original,
    items,
    players_in_team,
    total_distance,
    headshotKills_over_kills,
    killPlace_over_maxPlace,
    walkDistance_over_heals,
    walkDistance_over_kills,
    teamwork
])
run_experiments([
    original,
    kmeans_3_clusters, 
    kmeans_3_centroids,
    kmeans_4_clusters, 
    kmeans_4_centroids,
    kmeans_5_clusters, 
    kmeans_5_centroids
])
run_experiments([
    original,
    min_by_team,
    max_by_team,
    sum_by_team,
    median_by_team,
    mean_by_team,
    rank_by_team,
])