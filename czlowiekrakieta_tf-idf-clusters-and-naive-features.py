import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline, make_union

from sklearn.preprocessing import StandardScaler, scale

from sklearn.linear_model import LogisticRegression



vectorizer = TfidfVectorizer(stop_words='english', max_features=2048).fit(df['description'])
words_pipe = make_pipeline(

    TfidfVectorizer(stop_words='english', max_features=2048),

    LogisticRegression(C=5)

)
descs2vec = vectorizer.transform(df['description'])
def avenue_or_street(x):

    

    if 'avenue' in x.lower() or ' ave' in x.lower():

        return 1

    if 'street' in x.lower() or 'st.' in x.lower() or ' st' in x.lower():

        return 0

    else:

        return -1



def transform(df):



    price = scale(df['price'].tolist())

    bedrooms = df['bedrooms'].as_matrix()

    baths = df['bathrooms'].as_matrix()

    nr_of_features = df['features'].apply(len).as_matrix()

    avn_str = df['display_address'].apply(avenue_or_street).as_matrix()

    prop_bed = scale(df['price'].as_matrix()/(1+df['bedrooms'].as_matrix()))

    prop_bath = scale(df['price'].as_matrix()/(1+df['bathrooms'].as_matrix()))

    descr_log_length = scale(df['description'].fillna('0').apply(lambda x: np.log(1+len(x))))

    descr_length = scale(df['description'].apply(len))

    nr_of_photos = df['photos'].apply(len).as_matrix()

    

    

    return np.hstack([avn_str[None].T, 

                   nr_of_features[None].T, 

                   baths[None].T, 

                   price[None].T, 

                   bedrooms[None].T, 

                   nr_of_photos[None].T,

                  prop_bed[None].T,

                  prop_bath[None].T,

                  descr_log_length[None].T,

                  descr_length[None].T])
from collections import defaultdict



def build_feature_tfidf(series, max_feats=512):

    

    feat_counts = defaultdict(int)



    for f in series:



        for feat in f:

            feat_counts[feat] += 1

            

    D = min(max_feats, len(feat_counts))

    feat_counts = dict(sorted(feat_counts.items(), key=lambda x: x[1], reverse=True)[:D])

    ind_dict = dict(zip(feat_counts.keys(), range(D)))

    

    idf = np.log(series.shape[0]/(1+np.asarray(list(feat_counts.values()))))

    matrix = np.zeros((series.shape[0], D))

    for i, f in enumerate(series):

        

        for feat in f:

            if feat in ind_dict:

                matrix[i, ind_dict[feat]] = 1/len(f)*idf[ind_dict[feat]]

                

            

    return matrix, ind_dict, idf



ind_dict, idf = build_feature_tfidf(df['features'])[1:]



def transform_to_tfidf(series, ind_dict, idf):

    

    matrix = np.zeros((series.shape[0], len(ind_dict)))

    for i, f in enumerate(series):

        

        for feat in f:

            

            if feat in ind_dict:

                matrix[i, ind_dict[feat]] = 1/len(f)*idf[ind_dict[feat]]

                

    return matrix
def cut_outliers(matrix, perc=[.5, 99.5]):

    

    for i in range(matrix.shape[1]):

        

        matrix[:, i] = np.clip(matrix[:, i], np.percentile(matrix[:, i], perc[0]), 

                               np.percentile(matrix[:, i], perc[1]))

        

    return matrix
def get_avg_prices_wrt_clusters(to_cluster, return_stats=False, **params):

    

    km = KMeans(**params)

    clusters = km.fit_predict(to_cluster)

    

    all_prices = df['price'].as_matrix().flatten()

    

    n_clust = km.get_params()['n_clusters']

    

    stats = []

    

    final = np.zeros(df.shape[0])

    

    for i in range(n_clust):

        

        wh = np.where(clusters==i)[0]

        prices = all_prices[wh]

        z = [np.mean(prices), np.std(prices), np.median(prices)] 

        

        final[wh] = (prices - z[0])/(1e-3+z[1])

        

        stats.append(z)

        

        

    return (final, clusters, stats) if return_stats else (final, clusters)

    
month = todate.dt.month

day = todate.dt.day

hour = todate.dt.hour
from sklearn.ensemble import RandomForestClassifier
def create_data(df, with_tfidf=False):

    fin, cls = get_avg_prices_wrt_clusters(cut_outliers(df[['latitude', 'longitude']].as_matrix()))

    data = np.hstack([transform(df), fin.reshape(-1,1), cls.reshape(-1,1), cut_outliers(df[['latitude', 'longitude']].as_matrix())])

    

    return data

    
import xgboost as xgb
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.03

    param['max_depth'] = 6

    param['silent'] = 1

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = 1

    param['subsample'] = 0.7

    param['colsample_bytree'] = 0.7

    param['seed'] = seed_val

    num_rounds = num_rounds



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest)

    return pred_test_y, model
from sklearn.preprocessing import LabelEncoder, label
test = pd.read_json('test.json')