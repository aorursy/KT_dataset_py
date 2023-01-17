# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split,StratifiedKFold

import lightgbm as lgb

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/creditcard.csv')
data.shape
data.describe()
data.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total_null', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))



missing_data(data)
data_feature = data.drop(['Class'],axis = 1)

data_label = data['Class']

data_feature = data_feature.values

data_label = data_label.values
from sklearn.preprocessing import MinMaxScaler

data_feature = MinMaxScaler().fit_transform(data_feature)



def ewm(data):

    shape_x,shape_y = data.shape

    

    k = 1 / np.log(shape_y)

    y = data.sum(axis=0)

    p = data / y

    #计算pij

    test=p*np.log(p)

    test=np.nan_to_num(test)

    #计算每种指标的信息熵

    ej=-k*(test.sum(axis=0))

    #计算每种指标的权重

    wi=(1-ej)/np.sum(1-ej)

    

    return wi

w = ewm(data_feature)

w

features = [col for col in data.columns if col not in['Class']]

fold_importance_df = pd.DataFrame()

fold_importance_df["feature"] = features

fold_importance_df["importance"] = w



plt.figure(figsize=(10,5))

sns.barplot(y="importance", x="feature", data=fold_importance_df.sort_values(by="importance",ascending=False))

plt.title('Features importance ')

plt.tight_layout()
data_feature = data.drop(['Class'],axis = 1)

data_label = data['Class']

data_feature = data_feature.values

data_label = data_label.values
# Train_test split

random_state = 42

X_train, X_test, y_train, y_test = train_test_split(data_feature, data_label, test_size = 0.20, random_state = random_state, stratify = data_label)
lgb_params = {

    "objective" : "binary",

    "metric" : "l2",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 6,  #13

    "learning_rate" : 0.01,

#     "bagging_freq": 5,

#     "bagging_fraction" : 0.5,#0.4

#     "feature_fraction" : 0.041, #0.05

#     "min_data_in_leaf": 100,  #80

#     "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    #"lambda_l1" : 5,

    #"lambda_l2" : 5,

    "bagging_seed" : random_state,

    "verbosity" : 1,

    "seed": random_state

}

trn_data = lgb.Dataset(X_train, label=y_train)

val_data = lgb.Dataset(X_test, label=y_test)

evals_result = {}

lgb_clf = lgb.train(lgb_params,

                        trn_data,

                        100000,

                        valid_sets = [trn_data, val_data],

                        early_stopping_rounds=3000,

                        verbose_eval = 1000,

                        evals_result=evals_result

                       )
features = [col for col in data.columns if col not in['Class']]

fold_importance_df = pd.DataFrame()

fold_importance_df["feature"] = features

fold_importance_df["importance"] = lgb_clf.feature_importance()



plt.figure(figsize=(10,5))

sns.barplot(y="importance", x="feature", data=fold_importance_df.sort_values(by="importance",ascending=False))

plt.title('Features importance ')

plt.tight_layout()
g = lgb.plot_tree(lgb_clf, tree_index=0, figsize=(40, 20), show_info=['split_gain'])

plt.show()
import shap



# load JS visualization code to notebook

shap.initjs()



# explain the model's predictions using SHAP values

# (same syntax works for LightGBM, CatBoost, and scikit-learn models)

explainer = shap.TreeExplainer(lgb_clf)

shap_values = explainer.shap_values(X_train)



# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value, shap_values[0,:], X_train[0,:])
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)



# for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):

#     features = [col for col in df_train.columns if col not in ['target', 'ID_code']]



#     X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']

#     X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']

            

#     trn_data = lgb.Dataset(X_t, label=y_t)

#     val_data = lgb.Dataset(X_valid, label=y_valid)

#     evals_result = {}

#     lgb_clf = lgb.train(lgb_params,

#                         trn_data,

#                         100000,

#                         valid_sets = [trn_data, val_data],

#                         early_stopping_rounds=3000,

#                         verbose_eval = 5000,

#                         evals_result=evals_result

#                        )

#     p_valid += lgb_clf.predict(X_valid)

#     yp += lgb_clf.predict(X_test)

# #     fold_importance_df = pd.DataFrame()

# #     fold_importance_df["feature"] = features

# #     fold_importance_df["importance"] = lgb_clf.feature_importance()

# #     fold_importance_df["fold"] = fold + 1

# #     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

#     oof['predict'][val_idx] = p_valid/N

#     val_score = roc_auc_score(y_valid, p_valid)

#     val_aucs.append(val_score)

    

#     predictions['fold{}'.format(fold+1)] = yp/N