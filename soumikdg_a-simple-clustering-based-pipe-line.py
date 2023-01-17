import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

path = "../input/"

print(os.listdir(path))
from sklearn.metrics import roc_auc_score

import os

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
%time

Train_df = pd.read_csv(path + 'train.csv')

Train_df.head()
%time

Test_df = pd.read_csv(path + 'test.csv')

Test_df.head()
n_clusters = 5

test_size = 0.3
y_df = Train_df['target']

Train_df.drop(columns=['target'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(Train_df, y_df, test_size=test_size, random_state=40)
len(X_train), len(y_train)
columns = [i for i in X_train.columns if i not in ['ID_code']]
def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
kmeans = KMeans(n_clusters=n_clusters, max_iter=1000).fit(X_train[columns])

X_train["clusters"] = kmeans.labels_
cluster_idxs = [X_train["clusters"] == i for i in range(5)]
import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.1,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 4,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1,

    'max_bin': 50,

}
predictors = []



for i in range(n_clusters):

    

    print("TRAINING MODEL FOR CLUSTER: {}".format(i))

    x_i = X_train[cluster_idxs[i]]

    y_i = y_train[cluster_idxs[i]]



    num_folds = 3

    features = [c for c in x_i.columns if c not in ['ID_code', 'clusters', 'target']]

    folds = KFold(n_splits=num_folds, random_state=44000)



    x_i = x_i[features]



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_i.values, y_i.values)):



        X_trai, y_trai = x_i.iloc[trn_idx][features], y_i.iloc[trn_idx]

        X_val, y_val = x_i.iloc[val_idx][features], y_i.iloc[val_idx]



        X_trai, y_trai = augment(X_trai.values, y_trai.values)

        X_trai = pd.DataFrame(X_trai)



        print("Fold idx:{}".format(fold_ + 1))

        trn_data = lgb.Dataset(X_trai, label=y_trai)

        val_data = lgb.Dataset(X_val, label=y_val)



        clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 3000)

        predictors.append(clf)
def get_predictions(kmeans, X_test, n_clusters, get_score=True):

    features = [c for c in X_test.columns if c not in ['ID_code', 'target']]

    X_test['cluster'] = kmeans.predict(X_test[features])

    test_idxs = [X_test['cluster'] == i for i in range(n_clusters)]

    X_test.drop(columns=['cluster'], inplace=True)

    preds = []

    true = []



    for i in range(n_clusters):

        x_te = X_test[test_idxs[i]]

        pred = predictors[i].predict(x_te[features])

        preds.append(pred)

        if get_score:

            y_te = y_test[test_idxs[i]]

            true.append(y_te.values)

    x = []

    y = []

    for i in preds:

        x = x + list(i)

    if get_score:

        for i in true:

            y = y + list(i)

        print(roc_auc_score( np.array(y), np.array(x)))

    

    return np.array(x)

    
get_predictions(kmeans, X_test, n_clusters = 5, get_score=True)