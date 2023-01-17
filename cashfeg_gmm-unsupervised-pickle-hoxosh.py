import pickle

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.mixture import GaussianMixture

from tqdm import tqdm_notebook



import warnings

warnings.filterwarnings('ignore')
with open("../input/unsupervised-clustering-190617-v4/gm_models_v4.pkl", "rb") as f:

    gm_list = pickle.load(f)
train = pd.read_csv('../input/instant-gratification/train.csv')

test = pd.read_csv('../input/instant-gratification/test.csv')



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
not_very_good_turtles = np.array([345])
new_very_good_turtles = []
reg_covar = 0.01

tol = 0.01

n_components = 6

np.random.seed(1196)
i = 345



# ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

train2 = train[train['wheezy-copper-turtle-magic']==i]

test2 = test[test['wheezy-copper-turtle-magic']==i]

idx1 = train2.index

idx2 = test2.index

train2.reset_index(drop=True,inplace=True)



# FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

train3 = sel.transform(train2[cols])

test3 = sel.transform(test2[cols])



# seed

train_test3 = np.concatenate([train3, test3], axis=0)

gm_uns = GaussianMixture(

    n_components=n_components, reg_covar=reg_covar, tol=tol, weights_init=[1/6]*6, 

    random_state=1196

)

_ = gm_uns.fit(train_test3)

probs = np.ones((n_components, 2))

probs[:, 0] += gm_uns.predict_proba(train3[train2['target'].values == 0]).sum(axis=0)

probs[:, 1] += gm_uns.predict_proba(train3[train2['target'].values == 1]).sum(axis=0)





train_test_cnt = pd.value_counts(gm_uns.predict(train_test3)).values



if probs.min(axis=1).max() <= 15 and train_test_cnt.min() >= 90:

    print("turtle", i)

    print(probs)

    print(train_test_cnt)

    gm_list[i] = gm_uns

    new_very_good_turtles.append(i)

# FIT

oof_preds = np.zeros(train.shape[0])

train_preds = np.zeros(train.shape[0])

test_preds = np.zeros(test.shape[0])



for i in tqdm_notebook(range(512)):

    

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx_train = train2.index

    idx_test = test2.index

    train2.reset_index(drop=True,inplace=True)



    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])



    # predict

    probs = np.ones((n_components, 2))

    train_clusters = gm_list[i].predict(train3)

    test_clusters = gm_list[i].predict(test3)

    

    cluster_table = np.zeros((6, 2))

    # create table

    for j in range(train2.shape[0]):

        cluster_table[train_clusters[j], train2['target'].values[j]] += 1

    # oof-pred train

    for j in range(train2.shape[0]):

        oof_preds[idx_train[j]] = (

            (cluster_table[train_clusters[j], 1] - train2['target'].values[j]) / 

            np.max([(cluster_table[train_clusters[j], :].sum() - 1), 1])

        )

        train_preds[idx_train[j]] = (cluster_table[train_clusters[j], 1] / 

            cluster_table[train_clusters[j], :].sum())

    # pred test

    for j in range(test2.shape[0]):

        test_preds[idx_test[j]] = (cluster_table[test_clusters[j], 1] / 

            cluster_table[test_clusters[j], :].sum())
roc_auc_score(train["target"], oof_preds)
roc_auc_score(train["target"], train_preds)
with open("gm_models_v5.pkl", "wb") as f:

    pickle.dump(gm_list, f)