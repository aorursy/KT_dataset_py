import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000 # Jupyter notebook backend restricts number of points in plot

import pandas as pd

import scipy as scp

import csv

import seaborn as sns



train_master = pd.read_csv('../input/train.csv')

test_master = pd.read_csv('../input/test.csv')

train_master.describe()
binary_columns = [s for s in list(train_master.columns.values) if '_bin' in s]

categorical_columns = [s for s in list(train_master.columns.values) if '_cat' in s]

non_continuous_feature_subs = ['_cat', '_bin', 'target', 'id']

continuous_columns = [s for s in list(train_master.columns.values) 

                      if all(x not in s for x in non_continuous_feature_subs)]

target_column = 'target'



ind_columns = [s for s in list(train_master.columns.values) if '_ind' in s]

car_columns = [s for s in list(train_master.columns.values) if '_car' in s]

calc_columns = [s for s in list(train_master.columns.values) if '_calc' in s]

reg_columns = [s for s in list(train_master.columns.values) if '_reg' in s]
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go



init_notebook_mode()



labels = ['1','0']

values = [(train_master[target_column]==1).sum(),(train_master[target_column]==0).sum()]

colors = ['#FEBFB3', '#E1396C']



trace = go.Pie(labels=labels, values=values,

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))



iplot([trace])
zero_list = []

one_list = []

for col in binary_columns:

    zero_list.append((train_master[col]==0).sum())

    one_list.append((train_master[col]==1).sum())



trace1 = go.Bar(

    x=binary_columns,

    y=zero_list ,

    name='0s count'

)

trace2 = go.Bar(

    x=binary_columns,

    y=one_list,

    name='1s count'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='Count of 1s and 0s in binary variables'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stacked-bar')
binary_corr_data = []

r = 0

for i in binary_columns:

    binary_corr_data.append([])

    for j in binary_columns:

        s = sum(train_master[i]^train_master[j])/float(len(train_master[i]))

        binary_corr_data[r].append(s)

    r+=1
trace = go.Heatmap(z=binary_corr_data, x=binary_columns, y=binary_columns, colorscale='Greys')

data=[trace]

iplot(data)
binary_target_corr_data = []

for i in binary_columns:

    s = sum(train_master[i]^train_master[target_column])/float(len(train_master[i]))

    binary_target_corr_data.append(s)
binary_target_corr_chart = [go.Bar(

    x=binary_columns,

    y=binary_target_corr_data

)]

iplot(binary_target_corr_chart)
value_list = []

missing_list = []

for col in continuous_columns:

    value_list.append((train_master[col]!=-1).sum())

    missing_list.append((train_master[col]==-1).sum())



trace1 = go.Bar(

    x=continuous_columns,

    y=value_list ,

    name='Actual Values'

)

trace2 = go.Bar(

    x=continuous_columns,

    y=missing_list,

    name='Missing Values'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='Count of missing values in continuous variables'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stacked-bar')
from sklearn.feature_selection import chi2, mutual_info_classif



minfo_target_to_continuous_features = mutual_info_classif(

    train_master[continuous_columns],train_master[target_column])



minfo_target_to_continuous_chart = [go.Bar(

    x=continuous_columns,

    y=minfo_target_to_continuous_features

)]

iplot(minfo_target_to_continuous_chart)
continuous_corr_data = train_master[continuous_columns].corr(method='pearson').as_matrix()



trace = go.Heatmap(z=continuous_corr_data, x=continuous_columns, 

                   y=continuous_columns, colorscale='Greys')

data=[trace]

iplot(data)
def gini(y, pred):

    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)

    g = 2 * metrics.auc(fpr, tpr) -1

    return g
from sklearn.model_selection import StratifiedKFold



n_splits = 10

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
import numpy as np

from sklearn import metrics



def encode_cat_features(train_df, test_df, cat_cols, target_col_name, smoothing=1):

    prior = train_df[target_col_name].mean()

    probs_dict = {}

    for c in cat_cols:

        probs = train_df.groupby(c, as_index=False)[target_col_name].mean()

        probs['counts'] = train_df.groupby(c, as_index=False)[target_col_name].count()[[target_col_name]]

        probs['smoothing'] = 1 / (1 + np.exp(-(probs['counts'] - 1) / smoothing))

        probs['enc'] = prior * (1 - probs['smoothing']) + probs['target'] * probs['smoothing']

        probs_dict[c] = probs[[c,'enc']]

    return probs_dict
'''

import lightgbm as lgb

import numpy as np

import pandas as pd



np.random.seed(3)

model_scores = {}



# Drop binary columns with almost all zeros. 

# Why now? Just follow along for now. We have a lot of experimentation to be done

train = train_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)

test = test_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)



# Drop calculated features

# But WHY??? 

# Because we are assuming that tree can generate any complicated function 

# of base features and calculated features add no more information

# Is this assumption valid? Results will tell

calc_columns = [s for s in list(train_master.columns.values) if '_calc' in s]

train = train.drop(calc_columns, axis=1)  

test = test.drop(calc_columns, axis=1)



# Get categorical columns for encoding later

categorical_columns = [s for s in list(train_master.columns.values) if '_cat' in s]

target_column = 'target'



# Replace missing values with NaN

train = train.replace(-1, np.nan)

test = test.replace(-1, np.nan)



# Initialize DS to store validation fold predictions

y_val_fold = np.empty(len(train))



# Initialize DS to store test predictions with aggregate model and individual models

y_test = np.zeros(len(test))

y_test_model_1 = np.zeros(len(test))

y_test_model_2 = np.zeros(len(test))

y_test_model_3 = np.zeros(len(test))



for fold_number, (train_ids, val_ids) in enumerate(

    folds.split(train.drop(['id',target_column], axis=1), 

                train[target_column])):

    

    X = train.iloc[train_ids]

    X_val = train.iloc[val_ids]

    X_test = test

    

    # Encode categorical variables using training fold

    encoding_dict = encode_cat_features(X, X_val, categorical_columns, target_column)

    

    for c, encoding in encoding_dict.items():

        X = pd.merge(X, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))

        X = X.drop(c, axis = 1)

        X = X.rename(columns = {'enc':'enc_'+c})

        

        X_test = pd.merge(X_test, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))

        X_test = X_test.drop(c, axis = 1)

        X_test = X_test.rename(columns = {'enc':'enc_'+c})

        

        X_val = pd.merge(X_val, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))

        X_val = X_val.drop(c, axis = 1)

        X_val = X_val.rename(columns = {'enc':'enc_'+c})

        

    # Seperate target column and remove id column from all

    y = X[target_column]

    X = X.drop(['id',target_column], axis=1)

    X_test = X_test.drop('id', axis=1)

    y_val = X_val[target_column]

    X_val = X_val.drop(['id',target_column], axis=1)

    

    # Upsample data in training folds

    ids_to_duplicate = pd.Series(y == 1)

    X = pd.concat([X, X.loc[ids_to_duplicate]], axis=0)

    y = pd.concat([y, y.loc[ids_to_duplicate]], axis=0)

    # Again Upsample (total increase becomes 4 times)

    X = pd.concat([X, X.loc[ids_to_duplicate]], axis=0)

    y = pd.concat([y, y.loc[ids_to_duplicate]], axis=0)

    

    # Shuffle after concatenating duplicate rows

    # We cannot use inbuilt shuffles since both dataframes have to be shuffled in sync

    shuffled_ids = np.arange(len(X))

    np.random.shuffle(shuffled_ids)

    X = X.iloc[shuffled_ids]

    y = y.iloc[shuffled_ids]

    

    # Feature Selection goes here

    # TODO

    

    # Define parameters of GBM as explained before for 3 trees

    params_1 = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': 'auc',

        'max_depth': 3,

        'learning_rate': 0.05,

        'feature_fraction': 1,

        'bagging_fraction': 1,

        'bagging_freq': 10,

        'verbose': 0

    }

    params_2 = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': 'auc',

        'max_depth': 4,

        'learning_rate': 0.05,

        'feature_fraction': 0.9,

        'bagging_fraction': 0.9,

        'bagging_freq': 2,

        'verbose': 0

    }

    params_3 = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': 'auc',

        'max_depth': 5,

        'learning_rate': 0.05,

        'feature_fraction': 0.3,

        'bagging_fraction': 0.7,

        'bagging_freq': 10,

        'verbose': 0

    }

    

    # Create appropriate format for training and evaluation data

    lgb_train = lgb.Dataset(X, y)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    

    # Create the 3 classifiers with 1000 rounds and a window of 100 for early stopping

    clf_1 = lgb.train(params_1,lgb_train, num_boost_round=1000,

                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)

    clf_2 = lgb.train(params_2,lgb_train, num_boost_round=1000,

                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)

    clf_3 = lgb.train(params_3,lgb_train, num_boost_round=1000,

                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)

    

    # Predict raw scores for validation ids

    # At each fold, 1/10th of the training data get scores

    y_val_fold[val_ids] = (clf_1.predict(X_val, raw_score=True)+

                           clf_2.predict(X_val, raw_score=True)+

                           clf_3.predict(X_val, raw_score=True)) / 3



    # Predict and average over folds, raw scores for test data

    y_test += (clf_1.predict(X_test, raw_score=True)+

               clf_2.predict(X_test, raw_score=True)+

               clf_3.predict(X_test, raw_score=True)) / (3*n_splits)

    y_test_model_1 += clf_1.predict(X_test, raw_score=True) / n_splits

    y_test_model_2 += clf_2.predict(X_test, raw_score=True) / n_splits

    y_test_model_3 += clf_3.predict(X_test, raw_score=True) / n_splits

    

    # Display fold predictions

    # Gini requires only order and therefore raw scores need not be scaled

    print("Fold %2d : %.9f" % (fold_number + 1, gini(y_val, y_val_fold[val_ids])))

    

# Display aggregate predictions

# Gini requires only order and therefore raw scores need not be scaled

print("Average score over all folds: %.9f" % gini(train_master[target_column], y_val_fold))

'''
'''

temp = y_test

# Scale the raw scores to range [0.0, 1.0]

temp = np.add(temp,abs(min(temp)))/max(np.add(temp,abs(min(temp))))



df = pd.DataFrame(columns=['id', 'target'])

df['id']=test_master['id']

df['target']=temp

df.to_csv('benchmark__0_283.csv', index=False, float_format="%.9f")

df.shape

'''