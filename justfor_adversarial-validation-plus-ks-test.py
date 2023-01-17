#Load packages

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, KFold
#Load data; drop target and ID's

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.drop(train[['ID_code', 'target']], axis=1, inplace=True)

test.drop(test[['ID_code']], axis=1, inplace=True)
#https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split/

from tqdm import tqdm_notebook as tqdm



#df_test = pd.read_csv(test_path)

#df_test.drop(['ID_code'], axis=1, inplace=True)

df_test = test.values



unique_samples = []

unique_count = np.zeros_like(df_test)

for feature in tqdm(range(df_test.shape[1])):

    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)

    unique_count[index_[count_ == 1], feature] += 1



# Samples which have unique values are real the others are fake

real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]



print(len(real_samples_indexes))

print(len(synthetic_samples_indexes))
#df_test_real = df_test[real_samples_indexes].copy()

test = test.iloc[real_samples_indexes].copy()
#Create label array and complete dataset

y1 = np.array([0]*train.shape[0])

y2 = np.array([1]*test.shape[0])

y = np.concatenate((y1, y2))



X_data = pd.concat([train, test])

X_data.reset_index(drop=True, inplace=True)
print(X_data.shape, train.shape, test.shape)
#Initialize splits&LGBM

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)



lgb_model = lgb.LGBMClassifier(max_depth=-1,

                                   n_estimators=500,

                                   learning_rate=0.01,

                                   objective='binary', 

                                   n_jobs=-1)

                                   

counter = 1
#Train 5-fold adversarial validation classifier

for train_index, test_index in skf.split(X_data, y):

    print('\nFold {}'.format(counter))

    X_fit, X_val = X_data.loc[train_index], X_data.loc[test_index]

    y_fit, y_val = y[train_index], y[test_index]

    

    lgb_model.fit(X_fit, y_fit, eval_metric='auc', 

              eval_set=[(X_val, y_val)], 

              verbose=100, early_stopping_rounds=10)

    counter+=1
param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

         'learning_rate': 0.006,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 27,

         "metric": 'auc',

         "verbosity": -1}



random_state = 42

param = {

    "objective" : "binary", "metric" : "auc", "boosting": 'gbdt', "max_depth" : -1, "num_leaves" : 13,

    "learning_rate" : 0.01, "bagging_freq": 5, "bagging_fraction" : 0.4, "feature_fraction" : 0.05,

    "min_data_in_leaf": 80, "min_sum_heassian_in_leaf": 10, "tree_learner": "serial", "boost_from_average": "false",

    "bagging_seed" : random_state, "verbosity" : 1, "seed": random_state

}

train_test = X_data

target=y

features = [c for c in train_test.columns if c not in ['ID_code', 'target']]
folds = KFold(n_splits=5, shuffle=True, random_state=15)

oof = np.zeros(len(train_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_test.values, target)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(train_test.iloc[trn_idx][features], label=target[trn_idx])

    val_data = lgb.Dataset(train_test.iloc[val_idx][features], label=target[val_idx])



    num_round = 30000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000,\

                    early_stopping_rounds = 1400)

    oof[val_idx] = clf.predict(train_test.iloc[val_idx][features], num_iteration=clf.best_iteration)
from sklearn import model_selection, preprocessing, metrics

metrics.roc_auc_score(target, oof)



# Initial Script : AUC 0.534110539775; 

# Original params - AUC 0.503084033525



#=> deleting the "fake" test data showed a clearer picture, train & test are similar
#Load more packages

from scipy.stats import ks_2samp

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
#Perform KS-Test for each feature from train/test. Draw its distribution. Count features based on statistics.

#Plots are hidden. If you'd like to look at them - press "Output" button.

hypothesisnotrejected = []

hypothesisrejected = []



for col in train.columns:

    statistic, pvalue = ks_2samp(train[col], test[col])

    if pvalue>=statistic:

        hypothesisnotrejected.append(col)

    if pvalue<statistic:

        hypothesisrejected.append(col)

        

    plt.figure(figsize=(8,4))

    plt.title("Kolmogorov-Smirnov test for train/test\n"

              "feature: {}, statistics: {:.5f}, pvalue: {:5f}".format(col, statistic, pvalue))

    sns.kdeplot(train[col], color='blue', shade=True, label='Train')

    sns.kdeplot(test[col], color='green', shade=True, label='Test')



    plt.show()
len(hypothesisnotrejected), len(hypothesisrejected)
print(hypothesisrejected)