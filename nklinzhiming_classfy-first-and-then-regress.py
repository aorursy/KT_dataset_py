import numpy as np

import pandas as pd

import datetime

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

import warnings

from sklearn.model_selection import GroupKFold

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn import tree

import xgboost

from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss

warnings.filterwarnings('ignore')

np.random.seed(4590)
debug=4
if debug == 1:

    nrows = 50000

    print("本次程序运行是测试debug用")

else:

    nrows = None

    print("本次程序运行是运行数据模型")
df_train = pd.read_csv('../input/train.csv',nrows=nrows)

df_test = pd.read_csv('../input/test.csv',nrows=nrows)

df_hist_trans = pd.read_csv('../input/historical_transactions.csv',nrows=nrows)

df_new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv',nrows=nrows)
for df in [df_train,df_test,df_hist_trans,df_new_merchant_trans]:

    print(df.shape)
for df in [df_hist_trans,df_new_merchant_trans]:

    df['category_2'].fillna(1.0,inplace=True)

    df['category_3'].fillna('A',inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
def get_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
for df in [df_hist_trans,df_new_merchant_trans]:

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})

    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 

    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']
%%time

aggs = {}

for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

    aggs[col] = ['nunique']



aggs['purchase_amount'] = ['sum','max','min','mean','var','count']

aggs['installments'] = ['sum','max','min','mean','var']

aggs['purchase_date'] = ['max','min']

aggs['month_lag'] = ['max','min','mean','var']

aggs['month_diff'] = ['mean']

aggs['authorized_flag'] = ['sum', 'mean']

aggs['weekend'] = ['sum', 'mean']

aggs['category_1'] = ['sum', 'mean']

aggs['card_id'] = ['size']



for col in ['category_2','category_3']:

    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')

    aggs[col+'_mean'] = ['mean']    



new_columns = get_new_columns('hist',aggs)

df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)

df_hist_trans_group.columns = new_columns

df_hist_trans_group.reset_index(drop=False,inplace=True)

df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days

df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']

df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
%%time

aggs = {}

for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']

aggs['installments'] = ['sum','max','min','mean','var']

aggs['purchase_date'] = ['max','min']

aggs['month_lag'] = ['max','min','mean','var']

aggs['month_diff'] = ['mean']

aggs['weekend'] = ['sum', 'mean']

aggs['category_1'] = ['sum', 'mean']

aggs['card_id'] = ['size']



for col in ['category_2','category_3']:

    df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')

    aggs[col+'_mean'] = ['mean']

    

new_columns = get_new_columns('new_hist',aggs)

df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)

df_hist_trans_group.columns = new_columns

df_hist_trans_group.reset_index(drop=False,inplace=True)

df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days

df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']

df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
del df_hist_trans;gc.collect()

del df_new_merchant_trans;gc.collect()

df_train.head(5)
%%time

df_train['outliers'] = df_train.target.apply(lambda x: 1 if x<=-30 else 0)
df_train['outliers'].value_counts()
%%time

for df in [df_train,df_test]:

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['month'] = df['first_active_month'].dt.month

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days

    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\

                     'new_hist_purchase_date_min']:

        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']

    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']



for f in ['feature_1','feature_2','feature_3']:

    order_label = df_train.groupby([f])['outliers'].mean()

    df_train[f] = df_train[f].map(order_label)

    df_test[f] = df_test[f].map(order_label)

df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers','outliers2']]

y_clf = df_train['outliers']
%%time



folds = GroupKFold(n_splits=3)

#train_features = [_f for _f in train.columns if _f not in excluded_clf_features]

train_features = df_train_columns

train = df_train

test = df_test



print("train_features数量：",len(train_features))

importances = pd.DataFrame()

oof_clf_preds1 = np.zeros(train.shape[0])

oof_clf_preds11 = np.zeros(train.shape[0])

oof_clf_preds21 = np.zeros(train.shape[0])

sub_clf_preds1 = np.zeros(test.shape[0])

sub_clf_preds11 = np.zeros(test.shape[0])

sub_clf_preds21 = np.zeros(test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(df_train, y_clf, groups=train['card_id'])):

    print("folder----------",fold_)

    trn_x, trn_y = train[train_features].iloc[trn_], y_clf.iloc[trn_]

    val_x, val_y = train[train_features].iloc[val_], y_clf.iloc[val_]

    

    clf = lgb.LGBMClassifier(

        objective='binary',

        learning_rate=0.004,

        n_estimators=50000,

        max_depth=6,

        eval_metric='auc',

        lambda_l1=1.852,

        lambda_l2=1.548,

        child_weight=5.243,

        split_gain=0.04221,

        num_leaves=13,

        random_state=1

    )

    print("-"* 10 + "LightGBM Training" + "-"* 10)

    clf.fit(

        trn_x, trn_y,

        eval_set=[(val_x, val_y)],

        early_stopping_rounds=200,

        verbose=200

    )



    xgb_params = {

        'objective': 'binary:logistic',

        'booster': 'gbtree',

        'eta': 0.004,

        'lambda_l1':0.4472,

        'lambda_l2':0.08835,

        'max_depth':44,

        'child_weight':5.331,

        'split_gain':0.0454,

        'num_leaves':23,

        'is_unbalance':True,

        'random_state': 27

    }

    

    print("-"* 10 + "Xgboost Training" + "-"* 10)

    xg = XGBClassifier(**xgb_params, n_estimators=200000)

    xg.fit(

        trn_x, trn_y,

        eval_set=[(val_x, val_y)],

        early_stopping_rounds=200,

        verbose=200

    )

    print("-"* 10 + "Catboost Training" + "-"* 10)

    cat = CatBoostClassifier(iterations=20000,

                             learning_rate=0.02,

                             eval_metric='AUC',

                             random_seed = 42,

                             metric_period = 50,

                             od_wait=20)

    cat.fit(

        trn_x, trn_y,

        eval_set=[(val_x, val_y)],

        use_best_model=True,

        early_stopping_rounds=200,

        verbose=200)

    

    ###此处仅显示lgb模型训练下的importance

    imp_lgb_df = pd.DataFrame()

    imp_lgb_df['feature3'] = train_features

    imp_lgb_df['gain3'] = clf.booster_.feature_importance(importance_type='gain')

    imp_lgb_df['fold3'] = fold_ + 1

    importances = pd.concat([importances, imp_lgb_df], axis=0, sort=False)

    

    oof_clf_preds1[val_] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]

    oof_clf_preds11[val_] = xg.predict_proba(val_x)[:, 1]

    oof_clf_preds21[val_] = cat.predict_proba(val_x)[:, 1]

    

    

    print(roc_auc_score(val_y, oof_clf_preds1[val_]))

    print(roc_auc_score(val_y, oof_clf_preds11[val_]))

    sub_clf_preds1 += clf.predict_proba(test[train_features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    sub_clf_preds11 += xg.predict_proba(test[train_features])[:, 1] / folds.n_splits

    sub_clf_preds21 += cat.predict_proba(test[train_features])[:, 1] / folds.n_splits

    

print("LGB AUC模型分数：",roc_auc_score(y_clf, oof_clf_preds1))

print("XGB AUC模型分数：",roc_auc_score(y_clf, oof_clf_preds11))

print("CAT AUC模型分数：",roc_auc_score(y_clf, oof_clf_preds21))

import warnings

warnings.simplefilter('ignore', FutureWarning)

mean_gain = importances[['gain3', 'feature3']].groupby('feature3').mean()

importances['mean_gain3'] = importances['feature3'].map(mean_gain['gain3'])

plt.figure(figsize=(10, 16))

sns.barplot(x='gain3', y='feature3', data=importances.sort_values('mean_gain3', ascending=False).iloc[:350])

plt.savefig('lgbm_clf_importances13.png')
df_train['probal_lgb'] = oof_clf_preds1

df_train['probal_xgb'] = oof_clf_preds11

df_train['probal_cat'] = oof_clf_preds21



df_test['probal_lgb'] = sub_clf_preds1

df_test['probal_xgb'] = sub_clf_preds11

df_test['probal_cat'] = sub_clf_preds21

df_train.head()
lgb_min = np.min(df_train['probal_lgb'].values)

lgb_max = np.max(df_train['probal_lgb'].values)

xgb_min = np.min(df_train['probal_xgb'].values)

xgb_max = np.max(df_train['probal_xgb'].values)

cat_min = np.min(df_train['probal_cat'].values)

cat_max = np.max(df_train['probal_cat'].values)

print(lgb_min)

print(lgb_max)
df_train['probal_lgb'] = df_train.probal_lgb.apply(lambda x:(x-lgb_min)/(lgb_max-lgb_min))

df_train['probal_xgb'] = df_train.probal_xgb.apply(lambda x:(x-xgb_min)/(xgb_max-xgb_min))

df_train['probal_cat'] = df_train.probal_cat.apply(lambda x:(x-cat_min)/(cat_max-cat_min))



df_train.head()
df_train['lgb_xgb'] = oof_clf_preds1 + oof_clf_preds11

df_train['lgb_cat'] = oof_clf_preds1 + oof_clf_preds21

df_train['xgb_cat'] = oof_clf_preds11 + oof_clf_preds21

df_train['probal_combine'] = oof_clf_preds1 + oof_clf_preds11 + oof_clf_preds21

df_test['lgb_xgb'] = sub_clf_preds1 + sub_clf_preds11

df_test['lgb_cat'] = sub_clf_preds1 + sub_clf_preds21

df_test['xgb_cat'] = sub_clf_preds11 + sub_clf_preds21

df_test['probal_combine'] = sub_clf_preds1 + sub_clf_preds11 + sub_clf_preds21
df_train['probal_max'] = df_train[['probal_lgb','probal_xgb','probal_cat']].max(axis=1)

df_train['probal_min'] = df_train[['probal_lgb','probal_xgb','probal_cat']].min(axis=1)

df_test['probal_max'] = df_test[['probal_lgb','probal_xgb','probal_cat']].max(axis=1)

df_test['probal_min'] = df_test[['probal_lgb','probal_xgb','probal_cat']].min(axis=1)
df_train['larger2'] = df_train['probal_lgb'] + df_train['probal_xgb'] + df_train['probal_cat'] - df_train['probal_min']

df_train['less2'] = df_train['probal_lgb'] + df_train['probal_xgb'] + df_train['probal_cat'] - df_train['probal_max']

df_test['larger2'] = df_test['probal_lgb'] + df_test['probal_xgb'] + df_test['probal_cat'] - df_test['probal_min']

df_test['less2'] = df_test['probal_lgb'] + df_test['probal_xgb'] + df_test['probal_cat'] - df_test['probal_max']
for col in ['probal_lgb','probal_xgb','probal_cat','probal_max','probal_min',

            'lgb_xgb','lgb_cat','xgb_cat','probal_combine','larger2','less2']:

    df_train[col + "_multiplication"] = -33.219281 * df_train[col]

    df_test[col + "_multiplication"] = -33.219281 * df_test[col]
df_train.head(6)
x=df_train[df_train['target']<=-11]

x=x.sort_values(by ='target',ascending=False)

x=x[['target','outliers','probal_lgb','probal_xgb','probal_cat']]

x
df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers','outliers2']]

target = df_train['target']

#del df_train['target']
%%time

param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.01,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4590}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 200)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = df_train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits



np.sqrt(mean_squared_error(oof, target))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions

if debug == 1:

    sub_df.to_csv("submission_debug.csv", index=False)

    print("本次程序运行是测试debug用")

else:

    sub_df.to_csv("submission_model.csv", index=False)

    print("本次程序运行是运行数据模型")

df_train2 = df_train[df_train['outliers'] == 0]

target = df_train2['target']

del df_train2['target']
%%time

param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.01,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4590}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)

oof = np.zeros(len(df_train2))

predictions = np.zeros(len(df_test))

feature_importance_df2 = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train2,df_train2['outliers'].values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(df_train2.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train2.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 200)

    oof[val_idx] = clf.predict(df_train2.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    

    fold_importance_df2 = pd.DataFrame()

    fold_importance_df2["Feature2"] = df_train_columns

    fold_importance_df2["importance2"] = clf.feature_importance()

    fold_importance_df2["fold2"] = fold_ + 1

    feature_importance_df2 = pd.concat([feature_importance_df2, fold_importance_df2], axis=0)

    

    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits



np.sqrt(mean_squared_error(oof, target))
cols = (feature_importance_df2[["Feature2", "importance2"]]

        .groupby("Feature2")

        .mean()

        .sort_values(by="importance2", ascending=False)[:1000].index)



best_features = feature_importance_df2.loc[feature_importance_df2.Feature2.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance2",

            y="Feature2",

            data=best_features.sort_values(by="importance2",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances2.png')
sub_df2 = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df2["target"] = predictions

if debug == 1:

    sub_df2.to_csv("submission_debug2_without_outliers.csv", index=False)

    print("本次程序运行是测试debug用")

else:

    sub_df2.to_csv("submission_model2_without_outliers.csv", index=False)

    print("本次程序运行是运行数据模型")
