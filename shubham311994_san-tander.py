# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns

sns.set_style()

from sklearn.ensemble import RandomForestClassifier

from xgboost  import XGBClassifier

from sklearn.naive_bayes import ComplementNB,BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

import lightgbm as lgb





from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc, roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

t_d = test_data.copy()

test_data.drop(columns=['ID_code'],axis=1,inplace=True)
data.columns
# distribution of targets

colors = ['lightgreen','maroon']

plt.figure(figsize=(6,6))

plt.pie(data["target"].value_counts(), explode=(0, 0.25), labels= ["0", "1"], startangle=45, autopct='%1.1f%%', colors=colors)

plt.axis('equal')

plt.show()
all_data = data.copy()

Y = all_data['target']

all_data.drop(columns=['target','ID_code'],inplace=True)

X = all_data.loc[:,:]
### Creating new Features

def creating_features(X):

    square_df = X.apply(lambda a:a**2)

    cube_df =X.apply(lambda x:x**3)

    four_df = X.apply(lambda x:x**4)

    five_df = X.apply(lambda x:x**5)

    b = pd.concat([X,square_df,cube_df,four_df,five_df],axis=1)

    # cube_root = X.apply(lambda x:x**1/3)

    return b
def preprocess(data):

    ## fitting it on whole data first

    X = preprocessing.normalize(data)

    X = preprocessing.scale(X)

    # X_data_pca = preprocessing.normalize(X_data_pca)

    # X_data_pca = preprocessing.scale(X_data_pca)

    return X
def pca_obj(data,n):

    pca_obj_1 = PCA(n_components=n)

    X = pca_obj_1.fit_transform(data)

    return X
# X = creating_features(X)

X = preprocess(X)

# X_pca_data = pca_obj(X)

# X_pca_data = preprocess(X_pca_data)

# test_data = creating_features(test_data)

test_data = preprocess(test_data)

# test_data_pca=pca_obj(test_data)

# test_data_pca = preprocess(test_data_pca)
# Data augmentation

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
X_df = pd.DataFrame(X)

# X_df
random_state=32
train_data_1 = pd.concat([X_df,Y],axis=1)
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=random_state)

oof = data[['ID_code', 'target']]

oof['predict'] = 0

predictions = t_d[['ID_code']]

val_aucs = []

feature_importance = pd.DataFrame()
features = [col for col in data.columns if col not in ['target', 'ID_code']]

X_test = test_data.copy()
# Model parameters

lgb_params = {

    "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 31,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.4,

    "feature_fraction" : 0.05,

    "min_data_in_leaf": 150,

    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    "bagging_seed" : random_state,

    "verbosity" : 1,

    "seed": random_state}
for fold, (trn_idx, val_idx) in enumerate(skf.split(Y,Y)):

    X_train, y_train = X[trn_idx, :], Y[trn_idx]

    X_valid, y_valid = X[val_idx, :], Y[val_idx]

    

    N = 3

    p_valid,yp = 0,0

    for i in range(N):

        X_t, y_t = augment(X_train, y_train)

        X_t = pd.DataFrame(X_t)

        X_t = X_t.add_prefix('var_')

    

        trn_data = lgb.Dataset(X_t, label=y_t)

        val_data = lgb.Dataset(X_valid, label=y_valid)

        evals_result = {}

        lgb_clf = lgb.train(lgb_params,

                        trn_data,

                        100000,

                        valid_sets = [trn_data, val_data],

                        early_stopping_rounds=3000,

                        verbose_eval = 1000,

                        evals_result=evals_result

                       )

        p_valid += lgb_clf.predict(X_valid)

        yp += lgb_clf.predict(X_test)

    fold_importance = pd.DataFrame()

    fold_importance["feature"] = features

    fold_importance["importance"] = lgb_clf.feature_importance()

    fold_importance["fold"] = fold + 1

    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    oof['predict'][val_idx] = p_valid/N

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    

    predictions['fold{}'.format(fold+1)] = yp/N
# Submission

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)

predictions.to_csv('lgb_all_predictions.csv', index=None)

sub = pd.DataFrame({"ID_code":test["ID_code"].values})

sub["target"] = predictions['target']

sub.to_csv("lgb_submission.csv", index=False)

oof.to_csv('lgb_oof.csv', index=False)