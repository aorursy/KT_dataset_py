seed_random = 42

window_sizes = [10, 50]
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from sklearn.utils import shuffle

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer

import lightgbm as lgb

import xgboost as xgb

from pykalman import KalmanFilter

from functools import partial

import scipy as sp

import time

import datetime

import gc

from sklearn.tree import DecisionTreeClassifier

import shap
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        if col != 'time':

            col_type = df[col].dtypes

            if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

%%time

for window in window_sizes:

    train["rolling_mean_" + str(window)] = train['signal'].rolling(window=window).mean()

    train["rolling_std_" + str(window)] = train['signal'].rolling(window=window).std()

    train["rolling_var_" + str(window)] = train['signal'].rolling(window=window).var()

    train["rolling_min_" + str(window)] = train['signal'].rolling(window=window).min()

    train["rolling_max_" + str(window)] = train['signal'].rolling(window=window).max()

    

    #train["rolling_min_max_ratio_" + str(window)] = train["rolling_min_" + str(window)] / train["rolling_max_" + str(window)]

    #train["rolling_min_max_diff_" + str(window)] = train["rolling_max_" + str(window)] - train["rolling_min_" + str(window)]

    

    a = (train['signal'] - train['rolling_min_' + str(window)]) / (train['rolling_max_' + str(window)] - train['rolling_min_' + str(window)])

    train["norm_" + str(window)] = a * (np.floor(train['rolling_max_' + str(window)]) - np.ceil(train['rolling_min_' + str(window)]))

    

train = train.replace([np.inf, -np.inf], np.nan)    

train.fillna(0, inplace=True)
%%time

for window in window_sizes:

    test["rolling_mean_" + str(window)] = test['signal'].rolling(window=window).mean()

    test["rolling_std_" + str(window)] = test['signal'].rolling(window=window).std()

    test["rolling_var_" + str(window)] = test['signal'].rolling(window=window).var()

    test["rolling_min_" + str(window)] = test['signal'].rolling(window=window).min()

    test["rolling_max_" + str(window)] = test['signal'].rolling(window=window).max()

    

    #test["rolling_min_max_ratio_" + str(window)] = test["rolling_min_" + str(window)] / test["rolling_max_" + str(window)]

    #test["rolling_min_max_diff_" + str(window)] = test["rolling_max_" + str(window)] - test["rolling_min_" + str(window)]



    

    a = (test['signal'] - test['rolling_min_' + str(window)]) / (test['rolling_max_' + str(window)] - test['rolling_min_' + str(window)])

    test["norm_" + str(window)] = a * (np.floor(test['rolling_max_' + str(window)]) - np.ceil(test['rolling_min_' + str(window)]))



test = test.replace([np.inf, -np.inf], np.nan)    

test.fillna(0, inplace=True)
%%time

def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    

    # add shifts_1

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan

    

    # add shifts_2 - my upgrade

    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])

    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0] + [1]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==1].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-2'][i] = np.nan

    for i in df[df['batch_index']==49998].index:

        df['signal_shift_-2'][i] = np.nan

    

    df = df.drop(columns=['batch', 'batch_index', 'batch_slices', 'batch_slices2'])



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels']]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    df = df.replace([np.inf, -np.inf], np.nan)    

    df.fillna(0, inplace=True)

    gc.collect()

    return df



train = features(train)

test = features(test)
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
y = train['open_channels']

col = [c for c in train.columns if c not in ['time', 'open_channels', 'group', 'medianbatch', 'abs_avgbatch', 'abs_maxbatch']]
train.head()
# Thanks to https://www.kaggle.com/siavrez/simple-eda-model

def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)
%%time

# Thanks to https://www.kaggle.com/jazivxt/physically-possible with tuning from https://www.kaggle.com/siavrez/simple-eda-model and my tuning

X_train, X_valid, y_train, y_valid = train_test_split(train[col], y, test_size=0.01, random_state=seed_random)



model=lgb.LGBMClassifier(n_estimators=10)

model.fit(X_train,y_train)
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgb.plot_importance(model,ax = axes,height = 0.5,importance_type='split')

plt.show();plt.close()

gc.collect()
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgb.plot_importance(model,ax = axes,height = 0.5,importance_type='gain')

plt.show();plt.close()

gc.collect()
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=1).fit(X_valid, y_valid)



eli5.show_weights(perm, feature_names = X_valid.columns.tolist(), top=150)

features=X_valid.columns.tolist()

tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_train, y_train)

from sklearn import tree

import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)

graphviz.Source(tree_graph)
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_valid.iloc[:10000], model_features=features, feature='minbatch_msignal')



# plot it

pdp.pdp_plot(pdp_goals, 'minbatch_msignal')

plt.show()
pdp.pdp_plot(pdp_goals, 'stdbatch')

plt.show()
features_to_plot = ['minbatch_msignal', 'stdbatch']

inter1  =  pdp.pdp_interact(model=tree_model, dataset=X_valid.iloc[:10000], model_features=features, features=features_to_plot,)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour',which_classes=[0,1])

plt.show()



explainer = shap.TreeExplainer(model)

shap_values=explainer.shap_values(X_valid)

shap.summary_plot(shap_values, X_valid)
row_to_show = 1

data_for_prediction = X_valid.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





shap_values = explainer.shap_values(data_for_prediction_array)

shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[1], data_for_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

shap_values=explainer.shap_values(X_valid)

for name in ['minbatch_msignal',"stdbatch","rangebatch"]:

    shap.dependence_plot(name,shap_values[1],X_valid)