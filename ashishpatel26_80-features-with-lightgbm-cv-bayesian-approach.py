import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
# plt.style.use("fivethirtyeight")
plt.style.use('ggplot')
import seaborn as sns
import gc

sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
import datetime

import plotly.offline as ply
ply.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
# Read in the dataframes
def load_data():
    train = pd.read_csv('../input/train.csv',parse_dates=["first_active_month"])
    test = pd.read_csv('../input/test.csv',parse_dates=["first_active_month"])
    merchant = pd.read_csv('../input/merchants.csv')
    hist_trans = pd.read_csv('../input/historical_transactions.csv')
    print('train shape', train.shape)
    print('test shape', train.shape)
    print('merchants shape', merchant.shape)
    print('historical_transactions', hist_trans.shape)
    return (train,test,merchant,hist_trans)

gc.collect()
######### Function##################
def mis_value_graph(data, name = ""):
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = name,
        textfont=dict(size=20),
        marker=dict(
        color= generate_color(),
        line=dict(
            color='#000000',
            width=1,
        ), opacity = 0.85
    )
    ),
    ]
    layout= go.Layout(
        title= 'Total Missing Value of'+ str(name),
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig= go.Figure(data=data, layout=layout)
    ply.iplot(fig, filename='skin')
    
def datatypes_pie(data, title = ""):
    # Create a trace
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']
    trace1 = go.Pie(
        labels = ['float64','Int64'],
        values = data.dtypes.value_counts(),
        textfont=dict(size=20),
        marker=dict(colors=colors,line=dict(color='#000000', width=2)), hole = 0.45)
    layout = dict(title = "Data Types Count Percentage of "+ str(title))
    data = [trace1]
    ply.iplot(dict(data=data, layout=layout), filename='basic-line')
    

def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


import random

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
%%time
train,test,merchant, hist_trans = load_data()
train.name,test.name,merchant.name, hist_trans.name = "train","test","merchant", "hist_trans"
gc.collect()
%%time
for i in [train,test, merchant, hist_trans]:
    print("Data Types Cont of ",i.name)
    display(i.dtypes.value_counts())
    datatypes_pie(i, title = i.name)
    
gc.collect()
%%time
for i in [train,test,merchant, hist_trans]:
    print("Missing Value Count of ",i.name)
    mis_value_graph(i, name = i.name)
    
gc.collect()
# %%time
# for i in [train,test,merchant, hist_trans]:
#     print("Impute the Missing value of ", i.name)
#     mis_impute(i)
#     print("Done Imputation on", i.name)

# gc.collect()
test.shape
# train = train[train['target'] > -33]
# train.shape
%%time
x = train.target
data = [go.Histogram(x=x,
                     histnorm='probability')]
layout = go.Layout(
    title='Target Distribution',
    xaxis=dict(title='Value'),yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
ply.iplot(fig, filename='normalized histogram')
gc.collect()
x = train['first_active_month'].dt.date.value_counts()
x = x.sort_index()
data0 = [go.Histogram(x=x.index,y = x.values,histnorm='probability', marker=dict(color=generate_color()))]
layout = go.Layout(
    title='First active month count Train Data',
    xaxis=dict(title='First active month',ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of cards',ticklen=5, gridwidth=2),
    bargap=0.1,
    bargroupgap=0.2
)
fig = go.Figure(data=data0, layout=layout)
ply.iplot(fig, filename='normalized histogram')
gc.collect()
%%time
##---------------Time based Feature
train['day']= train['first_active_month'].dt.day 
train['dayofweek']= train['first_active_month'].dt.dayofweek
train['dayofyear']= train['first_active_month'].dt.dayofyear
train['days_in_month']= train['first_active_month'].dt.days_in_month
train['daysinmonth']= train['first_active_month'].dt.daysinmonth 
train['month']= train['first_active_month'].dt.month
train['week']= train['first_active_month'].dt.week 
train['weekday']= train['first_active_month'].dt.weekday
train['weekofyear']= train['first_active_month'].dt.weekofyear
train['year']= train['first_active_month'].dt.year
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days

      
##---------------Time based Test Feature      
test['day']= test['first_active_month'].dt.day 
test['dayofweek']= test['first_active_month'].dt.dayofweek
test['dayofyear']= test['first_active_month'].dt.dayofyear
test['days_in_month']= test['first_active_month'].dt.days_in_month
test['daysinmonth']= test['first_active_month'].dt.daysinmonth 
test['month']= test['first_active_month'].dt.month
test['week']= test['first_active_month'].dt.week 
test['weekday']= test['first_active_month'].dt.weekday
test['weekofyear']= test['first_active_month'].dt.weekofyear
test['year']= test['first_active_month'].dt.year
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days


print('train shape', train.shape)
print('test shape', test.shape)
gccollect()
%%time
#-----------------One-hot encode features
feat1 = pd.get_dummies(train['feature_1'], prefix='f1_')
feat2 = pd.get_dummies(train['feature_2'], prefix='f2_')
feat3 = pd.get_dummies(train['feature_3'], prefix='f3_')
feat4 = pd.get_dummies(test['feature_1'], prefix='f1_')
feat5 = pd.get_dummies(test['feature_2'], prefix='f2_')
feat6 = pd.get_dummies(test['feature_3'], prefix='f3_')

##---------------Numerical representation of the first active month
train = pd.concat([train,feat1, feat2, feat3], axis=1, sort=False)
test = pd.concat([test,feat4, feat5, feat6], axis=1, sort=False)

#shape of data
print('train shape', train.shape)
print('test shape', test.shape)
gc.collect()
%%time
hist_trans = pd.get_dummies(hist_trans, columns=['category_2', 'category_3'])
hist_trans['authorized_flag'] = hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})
hist_trans['category_1'] = hist_trans['category_1'].map({'Y': 1, 'N': 0})
hist_trans.head()
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)
gc.collect()
%%time
def aggregate_transactions(trans, prefix):  
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)
gc.collect()
%%time
merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
train = pd.merge(train, merch_hist, on='card_id',how='left')
test = pd.merge(test, merch_hist, on='card_id',how='left')
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)
gc.collect()
train.head()
trace0 = go.Box(y=train.feature_1,name="feature_1", marker=dict(color=generate_color()))
trace1 = go.Box(y=train.feature_2,name="feature_2", marker=dict(color=generate_color()))
trace2 = go.Box(y=train.feature_3,name="feature_3", marker=dict(color=generate_color()))
data = [trace0, trace1, trace2]
layout = go.Layout(
    title='Feature Boxplot Train',
    xaxis=dict(title='Value'),yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
ply.iplot(fig)

test.head()
trace0 = go.Box(y=test.feature_1,name="feature_1", marker=dict(color=generate_color()))
trace1 = go.Box(y=test.feature_2,name="feature_2", marker=dict(color=generate_color()))
trace2 = go.Box(y=test.feature_3,name="feature_3", marker=dict(color=generate_color()))
data = [trace0, trace1, trace2]
layout = go.Layout(
    title='Feature Boxplot Test',
    xaxis=dict(title='Value'),yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
ply.iplot(fig)
gc.collect()
%%time
new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")
display(new_trans_df.head())
new_trans_df.hist(figsize = (17,12))
gc.collect()
%%time
new_trans_df = pd.get_dummies(new_trans_df, columns=['category_2', 'category_3'])
new_trans_df['authorized_flag'] = new_trans_df['authorized_flag'].map({'Y': 1, 'N': 0})
new_trans_df['category_1'] = new_trans_df['category_1'].map({'Y': 1, 'N': 0})
new_trans_df.head()
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)
gc.collect()
%%time
merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
train = pd.merge(train, merch_hist, on='card_id',how='left')
test = pd.merge(test, merch_hist, on='card_id',how='left')
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)
gc.collect()
target = train['target']
drops = ['card_id', 'first_active_month', 'target', 'date']
use_cols = [c for c in train.columns if c not in drops]
features = list(train[use_cols].columns)
train[features].head()
%%time
print('train shape', train.shape)
print('test shape', test.shape)
train_df = train.copy()
test_df = test.copy()

print('train shape', train_df.shape)
print('test shape', test_df.shape)
gc.collect()
# train.dtypes
correlation = train_df.corr()
plt.figure(figsize=(20,15))
# mask = np.zeros_like(correlation)
# mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, annot=True)
%%time
# train_df.isnull().sum()
train_X = train_df[features]
test_X = test_df[features]
train_y = target
gc.collect()
%%time
print("X_train : ",train_X.shape)
print("X_test : ",test_X.shape)
print("Y_train : ",train_y.shape)
gc.collect()
from sklearn.datasets import load_boston
from sklearn.model_selection import (cross_val_score, train_test_split, 
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import r2_score
 
from lightgbm.sklearn import LGBMRegressor

hyper_space = {'n_estimators': [1000, 1500, 2000, 2500],
               'max_depth':  [4, 5, 8, -1],
               'num_leaves': [15, 31, 63, 127],
               'subsample': [0.6, 0.7, 0.8, 1.0],
               'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
               'learning_rate' : [0.01,0.02,0.03]
              }

est = lgb.LGBMRegressor(n_jobs=-1, random_state=2018)
gs = GridSearchCV(est, hyper_space, scoring='r2', cv=4, verbose=1)
gs_results = gs.fit(train_X, train_y)
print("BEST PARAMETERS: " + str(gs_results.best_params_))
print("BEST CV SCORE: " + str(gs_results.best_score_))
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


lgb_params = {"objective" : "regression", "metric" : "rmse", 
               "max_depth": 7, "min_child_samples": 20, 
               "reg_alpha": 1, "reg_lambda": 1,
               "num_leaves" : 64, "learning_rate" : 0.01, 
               "subsample" : 0.8, "colsample_bytree" : 0.8, 
               "verbosity": -1}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_X))
predictions_lgb = np.zeros(len(test_X))

features_lgb = list(train_X.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_X)):
    trn_data = lgb.Dataset(train_X.iloc[trn_idx], label=train_y.iloc[trn_idx])
    val_data = lgb.Dataset(train_X.iloc[val_idx], label=train_y.iloc[val_idx])

    print("-" * 20 +"LGB Fold:"+str(fold_)+ "-" * 20)
    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)
    oof_lgb[val_idx] = clf.predict(train_X.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test_X, num_iteration=clf.best_iteration) / FOLDs.n_splits
    

print("Best RMSE: ",np.sqrt(mean_squared_error(oof_lgb, train_y)))
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

def run_lgb(train_X, train_y, val_X, val_y, test_X):
#     params = {
#         "objective" : "regression",
#         "metric" : "rmse",
#         "num_leaves" : 128,
#         'max_depth' : 7,
#         "min_child_weight" : 20,
#         "learning_rate" : 0.001,
#         "reg_alpha": 1, "reg_lambda": 1,
#         "learning_rate" : 0.01,
#         "subsample" : 0.8, "colsample_bytree" : 0.8, 
#         "verbose": 1
#     }
    params={'learning_rate': 0.01,
            'objective':'regression',
            'metric':'rmse',
            'num_leaves': 31,
            'verbose': 1,
            'bagging_fraction': 0.9,
            'feature_fraction': 0.9,
            "random_state":1,
#             'max_depth': 5,
#             "bagging_seed" : 42,
#             "verbosity" : -1,
#             "bagging_frequency" : 5,
#             'lambda_l2': 0.5,
#             'lambda_l1': 0.5,
#             'min_child_samples': 36
           }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
 

pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
for fold_, (dev_index, val_index) in enumerate(kf.split(train_df, train_y)):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    print("-" * 20 +"LGB Fold:"+str(fold_)+ "-" * 20)    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += model.predict(test_X, num_iteration=clf.best_iteration)
pred_test /= 5
###--------LightGBM1 feature Importance--------------
print("Feature Importance For LGB Model1")
cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

###--------LightGBM2 feature Importance--------------

print("Feature Importance For LGB Model2")
fig, ax = plt.subplots(figsize=(20,10))
lgb.plot_importance(model, max_num_features=50, height=0.9, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=25)
plt.show()
plt.savefig('lgbm_importances1.png')
temp_df = pd.DataFrame()
temp_df["target1"] = predictions_lgb
temp_df["target2"] = pred_test
temp_df["target3"] = temp_df["target1"] * 0.5 + temp_df["target2"] * 0.5
sub = pd.read_csv("../input/sample_submission.csv")
sub['target'] = temp_df["target3"]
sub.to_csv("ELO_LGB_Blend.csv", index=False)
sub.head()
sub = pd.read_csv("../input/sample_submission.csv")
sub['target'] = temp_df["target3"] * 0.6 + sub['target'] * 0.4
# sub['target'] = sub['target'].apply(lambda x : 0 if x < 0 else x)
sub.to_csv("ELO_LGB_Sample.csv", index=False)
sub.head()
# sub[sub['target'] == 0].count()
sub = pd.read_csv("../input/sample_submission.csv")
sub['target'] = temp_df["target1"] 
# sub['target'] = sub['target'].apply(lambda x : 0 if x < 0 else x)
sub.to_csv("ELO_LGB1.csv", index=False)
sub.head()
# sub[sub['target'] == 0].count()
sub = pd.read_csv("../input/sample_submission.csv")
sub['target'] = temp_df["target2"] 
# sub['target'] = sub['target'].apply(lambda x : 0 if x < 0 else x)
sub.to_csv("ELO_LGB2.csv", index=False)
sub.head()
# sub[sub['target'] == 0].count()