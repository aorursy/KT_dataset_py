# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV faya_2015





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale



from sklearn import model_selection

from sklearn.model_selection import train_test_split



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

%matplotlib inline
#Read data first

train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")

sample_df=pd.read_csv("../input/sample_submission.csv")
# This is a binary classification problem

train_df.target.value_counts()
#Lets look at the dimensions of the training data

train_df.shape
train_df.columns.values

# Looks like a bunch of anonymized variables
# Any missing values?

null_data = train_df[train_df.isnull().any(axis=1)]

null_data
#Let's do a heatmap to see the correlations

train_df_corr=train_df.corr()
train_df_corr.head()
sns.heatmap(train_df_corr)
#It does not look like any of the variables are correlated

upper = train_df_corr.abs().where(np.triu(np.ones(train_df_corr.abs().shape), k=1).astype(np.bool))



# Find features with correlation greater than 0.2

correlated = [column for column in upper.columns if any(upper[column] > 0.2)]



correlated



# no variables are correlated
# there are too many input columns, let's try dimensionality reduction

#1. PCA

#2. Important features using Random forest

#3. PCa using random forest
#convert it to numpy arrays

X=train_df.drop(columns=['ID_code','target']).values



#Scaling the values

X = scale(X)



pca = PCA(n_components=200)



pca.fit(X)



#The amount of variance that each PC explains

var= pca.explained_variance_ratio_



#Cumulative Variance explains

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)



print(var1)



plt.plot(var1)



#pd.DataFrame(pca.components_).round(2).to_csv("xyz.csv")



# now look at heatmap and figure out explainability
import lightgbm as lgb

param = {

        'num_leaves': 10,

        'max_bin': 119,

        'min_data_in_leaf': 11,

        'learning_rate': 0.02,

        'min_sum_hessian_in_leaf': 0.00245,

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'feature_fraction': 0.05,

        'lambda_l1': 4.972,

        'lambda_l2': 2.276,

        'min_gain_to_split': 0.65,

        'max_depth': 14,

        'save_binary': True,

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']
%%time

import time

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()



start = time.time()





for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / 5



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
##submission

sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("lgb_submission.csv", index=False)