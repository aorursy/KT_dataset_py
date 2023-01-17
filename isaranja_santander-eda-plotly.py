import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



import collections



pd.options.display.max_columns = 201



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# preprocessing

from sklearn.preprocessing import MinMaxScaler # normalization



# keras packages

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten



# model selection and metrics

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn import metrics



import seaborn as sns



from keras import optimizers



import tensorflow as tf



import lightgbm as lgb
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")

submission = pd.read_csv("../input/sample_submission.csv")



train_df = train_df.assign(isTrain = True)

test_df = test_df.assign(isTrain=False)

full_df = pd.concat([train_df,test_df],sort=False)
scaler = MinMaxScaler()

full_df.loc[:,'var_0':'var_199']=scaler.fit_transform(full_df.loc[:,'var_0':'var_199'].values)
full_df_bk =  full_df.copy()
full_df = full_df_bk.copy()

full_df = full_df.loc[~(np.round(full_df.var_0,4).isin(np.round(np.random.normal(0.575, 0.02, 100),4)))]

full_df = full_df.loc[~(np.round(full_df.var_1,4).isin(np.round(np.random.normal(0.9025, 0.02, 400),4)))]

full_df = full_df.loc[~(np.round(full_df.var_2,4).isin(np.round(np.random.normal(0.5925, 0.01, 100),4)))]

full_df = full_df.loc[~(np.round(full_df.var_108,4).isin(np.round(np.random.normal(0.465, 0.001, 50),4)))]

full_df = full_df.loc[~(np.round(full_df.var_81,4).isin(np.round(np.random.normal(0.1, 0.01, 400),4)))]

full_df = full_df.loc[~(np.round(full_df.var_99,4).isin(np.round(np.random.normal(0.9000, 0.01, 400),4)))]

df = full_df.loc[~(np.round(full_df.var_53,4).isin(np.round(np.random.normal(0.9075, 0.005, 400),4)))]
var = 'var_67'

trace1 = go.Histogram(

    x=df.loc[df.target==0,var].values,

    opacity=0.75

)

trace2 = go.Histogram(

    x=df.loc[df.target==1,var].values,

    opacity=0.75

)



data = [trace1, trace2]

layout = go.Layout(barmode = 'overlay')

fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename = 'overlaid histogram')
X = df.loc[:,'var_0':'var_199'].values

x = X[(df.isTrain)]

y = df[(df.isTrain)].target



param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1

}



folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(x, y)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(x[trn_idx][:], label=y[trn_idx])

    val_data = lgb.Dataset(x[val_idx][:], label=y[val_idx])



    num_round = 1000000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)

    oof[val_idx] = clf.predict(x[val_idx][:], num_iteration=clf.best_iteration)

     

    #predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.2,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.1,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1,

    'seed': 42

}



X = df.loc[:,'var_0':'var_199'].values

x = X[(df.isTrain)]

y = df[(df.isTrain)].target



x_train, x_val, y_train, y_val = train_test_split(x, y , test_size=0.33, stratify=y, random_state=42)



trn_data = lgb.Dataset(x_train, label=y_train)

val_data = lgb.Dataset(x_val, label=y_val)

clf = lgb.train(param, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),df.loc[:,'var_0':'var_199'].columns)), columns=['Value','Feature'])



plt.figure(figsize=(10, 30))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()
var_53

var_146

var_99

var_179

var_67