import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
%matplotlib inline
train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str})
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str})
train.shape, test.shape
train.head()
train.columns
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']
train.columns
train['target'] = y_reg
for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day
    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    df['next_session_1'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60

y_reg = train['target']
del train['target']
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]
for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])
folds = get_folds(df=train, n_splits=5)

train_features = [_f for _f in train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
train['predictions'] = np.expm1(oof_reg_preds)
test['predictions'] = sub_reg_preds
# Aggregate data at User level
trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
%%time
# Create a list of predictions for each Visitor
trn_pred_list = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
# Create a DataFrame with VisitorId as index
# trn_pred_list contains dict 
# so creating a dataframe from it will expand dict values into columns
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns
trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions
gc.collect()
full_data.shape
%%time
sub_pred_list = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
sub_data = test[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
for f in trn_feats:
    if f not in sub_all_predictions.columns:
        sub_all_predictions[f] = np.nan
sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
del sub_data, sub_all_predictions
gc.collect()
sub_full_data.shape
train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
df = pd.concat([full_data,sub_full_data],sort=False)
df = df.reset_index(drop=False)
for c in df.columns[1:]:
    if((df[c].min()>=0)&(df[c].max()>=10)):
        df[c] = np.log1p(df[c])
    elif((df[c].min()<0)&((df[c].max()-df[c].min())>=10)):
        df.loc[df[c]!=0,c] = np.sign(df.loc[df[c]!=0,c])*np.log(np.abs(df.loc[df[c]!=0,c]))
from sklearn.preprocessing import StandardScaler
for c in df.columns[1:]:
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]),c] = np.nan
    df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
df.fillna(-99999,inplace=True)
gp_trn_users = df[:full_data.shape[0]].copy().set_index('fullVisitorId')
gp_trn_users['target'] = np.log1p(trn_user_target['target'].values)
#gp_trn_users['target'] /= gp_trn_users['target'].max()
gp_sub_users = df[full_data.shape[0]:].copy().set_index('fullVisitorId')
newcols =  [x.replace('.','_') for x in gp_trn_users.columns]
gp_trn_users.columns = newcols
newcols =  [x.replace('.','_') for x in gp_sub_users.columns]
gp_sub_users.columns = newcols
#gp_trn_users.to_csv('gptrain.csv',index=False,float_format='%.6f')
def GP1(data):
    return (12.500000 +
            0.100000*np.tanh(((((((((((-3.0) + (data["t_sum_log"]))) - ((((12.95331287384033203)) / 2.0)))) * ((4.50129508972167969)))) + (data["t_sum_act"]))) * 2.0)) +
            0.100000*np.tanh(((((((((((((data["t_sum_log"]) - ((9.0)))) * 2.0)) + (data["channelGrouping"]))) * 2.0)) + (2.0))) * 2.0)) +
            0.100000*np.tanh((((10.0)) * (((data["t_sum_log"]) - (((((10.0)) + ((((4.60818958282470703)) + (((data["t_sum_log"]) - (data["t_sum_act"]))))))/2.0)))))) +
            0.100000*np.tanh(((((((((((data["t_sum_act"]) - ((8.0)))) * 2.0)) - (((3.0) * 2.0)))) + (data["t_sum_log"]))) * 2.0)) +
            0.100000*np.tanh((((((((((((((((data["t_sum_act"]) + (2.0))/2.0)) + (data["t_sum_log"]))) - ((12.46262645721435547)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((((10.0)) * (((((data["t_sum_log"]) + (((((data["t_mean"]) - (3.0))) - ((12.30329895019531250)))))) * 2.0)))) * 2.0)) +
            0.100000*np.tanh((((11.12195396423339844)) * (((((data["t_sum_log"]) - ((11.12195396423339844)))) + (((data["t_mean"]) / 2.0)))))) +
            0.100000*np.tanh(((((data["pred_2"]) + (data["pred_1"]))) - (((((data["pred_3"]) - ((7.0)))) * (((data["t_mean"]) - ((7.0)))))))) +
            0.100000*np.tanh((((((8.0)) - (data["next_session_2"]))) + (((((data["t_sum_log"]) - (np.maximum((((8.0))), ((data["next_session_2"])))))) * ((8.41708469390869141)))))) +
            0.100000*np.tanh(((((data["pred_3"]) + (((((data["pred_3"]) + (((((data["pred_2"]) - ((4.0)))) + (data["pred_4"]))))) * 2.0)))) * 2.0)) +
            0.100000*np.tanh(((((((((np.where(((data["t_sum_log"]) - ((8.0)))>0, data["t_sum_log"], data["t_sum_act"] )) - ((8.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((data["pred_3"]) + (((((((data["next_session_1"]) + (data["pred_0"]))/2.0)) + (data["pred_1"]))/2.0)))) - ((5.55097246170043945)))) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((((14.49165344238281250)) * (((data["t_sum_act"]) - ((7.80256223678588867)))))) + (np.where(data["pred_5"] < -99998, data["t_sum_log"], ((data["t_sum_log"]) * 2.0) )))) +
            0.100000*np.tanh(((((((data["t_sum_log"]) - ((6.95692586898803711)))) * (data["t_sum_act"]))) + ((((8.0)) * (((data["t_sum_act"]) - ((6.95692586898803711)))))))) +
            0.100000*np.tanh(((((((((((((((data["t_sum_act"]) - ((5.79598665237426758)))) * 2.0)) + (-3.0))) * 2.0)) * 2.0)) * 2.0)) + ((5.79598665237426758)))) +
            0.100000*np.tanh(((data["pred_2"]) + (((data["pred_4"]) + (((data["pred_2"]) + (((((data["t_median"]) + (((-3.0) * 2.0)))) * 2.0)))))))) +
            0.100000*np.tanh(((((((((((((data["t_sum_log"]) - ((7.0)))) - ((((data["t_sum_log"]) > (data["t_sum_act"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((data["t_sum_log"]) + (((((data["t_sum_log"]) + (((((data["t_sum_act"]) - ((11.79764175415039062)))) * 2.0)))) * 2.0)))) * ((11.79764175415039062)))) +
            0.100000*np.tanh(((((((((((data["t_sum_log"]) - ((8.0)))) * 2.0)) * 2.0)) + ((((6.84783124923706055)) - (data["visitNumber"]))))) * ((8.0)))) +
            0.100000*np.tanh((((((7.0)) * (((data["t_sum_log"]) + (((data["t_sum_act"]) - ((14.84496498107910156)))))))) + (-3.0))) +
            0.100000*np.tanh((((7.0)) * (((((((data["t_sum_act"]) - ((7.0)))) * 2.0)) + (np.tanh((((data["totals_pageviews"]) + (data["pred_4"]))))))))) +
            0.100000*np.tanh(((((((((((data["t_sum_log"]) - ((8.0)))) * 2.0)) * 2.0)) + (((data["t_mean"]) - (data["visitNumber"]))))) * 2.0)) +
            0.100000*np.tanh(((((((((np.minimum(((data["pred_0"])), ((data["pred_4"])))) + (((data["pred_3"]) - (3.0))))) * 2.0)) * 2.0)) + (data["pred_3"]))) +
            0.100000*np.tanh((((((9.0)) * (((((((data["t_sum_log"]) - ((9.88647079467773438)))) - (-3.0))) * 2.0)))) * 2.0)) +
            0.100000*np.tanh(((((data["t_sum_act"]) + (((((((data["t_sum_act"]) + (((((data["t_sum_act"]) - ((10.89448451995849609)))) * 2.0)))) * 2.0)) * 2.0)))) * 2.0)) +
            0.100000*np.tanh(((((((((((data["t_sum_log"]) - ((9.0)))) * 2.0)) * 2.0)) + ((2.0)))) + ((((9.0)) - (data["visitNumber"]))))) +
            0.100000*np.tanh(((((((data["t_mean"]) - ((((9.0)) - (((data["t_sum_log"]) - ((7.0)))))))) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((((((((((7.24431943893432617)) * (((data["t_sum_log"]) - ((7.24431943893432617)))))) - ((7.24431943893432617)))) + (-2.0))) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((((14.16644096374511719)) * (((data["t_sum_act"]) - ((((14.16644096374511719)) - (np.maximum((((8.0))), ((data["t_sum_log"])))))))))) - ((8.0)))) +
            0.100000*np.tanh(((((((((((data["t_mean"]) + (((((data["t_sum_log"]) - ((((5.0)) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((data["pred_4"]) + ((((14.16003227233886719)) * ((((14.16003227233886719)) * (((np.maximum(((data["t_median"])), ((data["t_mean"])))) - ((6.0)))))))))) +
            0.100000*np.tanh(((data["pred_1"]) + ((((((8.92317485809326172)) * (((np.maximum(((data["t_sum_log"])), ((data["pred_1"])))) - ((8.92317485809326172)))))) * 2.0)))) +
            0.100000*np.tanh((((((10.0)) * 2.0)) * (((data["t_sum_log"]) - ((((((10.0)) - (data["t_mean"]))) * 2.0)))))) +
            0.100000*np.tanh(((((((((data["t_sum_log"]) - ((8.0)))) * ((((9.0)) + ((12.95555210113525391)))))) - (data["t_sum_log"]))) * ((9.0)))) +
            0.100000*np.tanh(((((((data["pred_3"]) + (((((data["t_sum_log"]) - ((8.74195766448974609)))) * (((data["visitNumber"]) * 2.0)))))) * 2.0)) * ((8.74195766448974609)))) +
            0.100000*np.tanh((((((10.68931770324707031)) * (((((((((data["t_sum_log"]) - ((10.0)))) * 2.0)) + (data["t_mean"]))) * 2.0)))) + (data["t_sum_log"]))) +
            0.100000*np.tanh(((((data["t_sum_log"]) + (((((((data["t_sum_log"]) - (np.maximum((((9.0))), ((data["visitNumber"])))))) * 2.0)) * 2.0)))) * 2.0)) +
            0.100000*np.tanh((((10.33415031433105469)) * ((((14.93418216705322266)) * (((((((data["t_sum_act"]) - ((9.26103305816650391)))) * 2.0)) + (data["t_sum_log"]))))))) +
            0.100000*np.tanh(((((np.maximum(((data["t_sum_log"])), ((((data["totals_hits"]) + (data["pred_1"])))))) - (np.maximum((((9.0))), ((data["visitNumber"])))))) * ((9.0)))) +
            0.100000*np.tanh((((((((data["pred_0"]) + (data["t_sum_log"]))/2.0)) - ((((14.61157703399658203)) - (data["t_sum_log"]))))) + (data["pred_1"]))) +
            0.100000*np.tanh((((((((8.0)) - (data["t_mean"]))) * (-3.0))) - ((((7.0)) - (np.maximum(((data["pred_0"])), ((data["pred_3"])))))))) +
            0.100000*np.tanh((((((12.26031208038330078)) * (((data["t_mean"]) + (((data["t_sum_log"]) - ((12.26031208038330078)))))))) * 2.0)) +
            0.100000*np.tanh((((9.30737781524658203)) * (((((data["t_sum_log"]) - ((9.30737781524658203)))) + (((((data["t_sum_act"]) * 2.0)) - ((9.30737781524658203)))))))) +
            0.100000*np.tanh((((((((9.85758876800537109)) * (((data["t_sum_log"]) - ((8.81438827514648438)))))) - ((8.0)))) * ((8.81438827514648438)))) +
            0.100000*np.tanh(((((((((data["pred_0"]) - ((((13.48505210876464844)) - (((data["pred_1"]) + (data["t_sum_log"]))))))) * 2.0)) - (2.0))) * 2.0)) +
            0.100000*np.tanh((((((((((((9.53180027008056641)) * (((data["t_sum_log"]) - ((6.0)))))) * 2.0)) * 2.0)) + ((((9.53180027008056641)) / 2.0)))) * 2.0)) +
            0.100000*np.tanh(((data["pred_3"]) + (((data["pred_3"]) + (((data["pred_3"]) + (((((((data["t_sum_log"]) - ((10.0)))) * 2.0)) * 2.0)))))))) +
            0.100000*np.tanh(((((((((((((data["t_sum_act"]) - ((11.60270595550537109)))) + (data["t_sum_log"]))) * 2.0)) * ((11.60270595550537109)))) - ((11.60270595550537109)))) * 2.0)) +
            0.100000*np.tanh(((((((((((data["pred_0"]) - ((9.22513771057128906)))) * ((9.22513771057128906)))) - (data["t_median"]))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((data["pred_3"]) + (((((data["t_sum_log"]) - ((10.0)))) * ((10.0)))))) * 2.0)) + ((10.0)))) * 2.0)) +
            0.100000*np.tanh(((((((data["t_sum_act"]) - ((6.00677633285522461)))) * ((((6.00677633285522461)) - (((data["pred_3"]) - (data["t_sum_act"]))))))) + (data["t_sum_act"]))) +
            0.100000*np.tanh(((((data["pred_1"]) + (((((data["pred_1"]) + (((((data["t_mean"]) - ((14.10800457000732422)))) + (data["pred_0"]))))) * 2.0)))) * 2.0)) +
            0.100000*np.tanh((((((9.98417091369628906)) * 2.0)) * ((((8.10735034942626953)) * ((((9.98417091369628906)) * (((((data["pred_2"]) - ((4.0)))) * 2.0)))))))) +
            0.100000*np.tanh(((((((((((((data["t_sum_log"]) - ((9.48249053955078125)))) + (-2.0))) + (data["t_sum_act"]))) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((data["pred_2"]) - (np.maximum(((data["next_session_2"])), ((np.maximum(((((data["visitNumber"]) * (data["visitNumber"])))), ((data["visitNumber"]))))))))) * 2.0)) +
            0.100000*np.tanh(((((((((data["t_sum_act"]) + (((((((((data["t_sum_act"]) - ((6.93721342086791992)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((11.06935405731201172)) * (((((((data["t_sum_log"]) - ((11.06935405731201172)))) + (np.where(data["pred_1"]>0, 0.0, data["t_sum_log"] )))) * 2.0)))) +
            0.100000*np.tanh(((((np.where(((-3.0) + (data["pred_1"]))>0, (6.0), -3.0 )) + (data["pred_3"]))) - (data["visitNumber"]))) +
            0.100000*np.tanh((((((10.0)) * ((((((np.where(data["pred_1"]>0, (2.85201859474182129), data["pred_0"] )) + (data["pred_1"]))/2.0)) + (-3.0))))) * 2.0)) +
            0.100000*np.tanh(((((((((data["t_sum_log"]) + (((-3.0) * 2.0)))) * 2.0)) * 2.0)) * ((((8.0)) - (data["t_sum_log"]))))) +
            0.100000*np.tanh(((((((((((((data["t_sum_act"]) - (np.maximum((((6.0))), ((data["visitNumber"])))))) * 2.0)) * 2.0)) + (data["next_session_1"]))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((data["t_sum_act"]) - (((3.0) * 2.0)))) + ((0.43446072936058044)))) * ((9.0)))) * 2.0)) +
            0.100000*np.tanh((((9.72239112854003906)) * (((np.where(data["pred_2"] < -99998, data["pred_1"], data["t_sum_log"] )) - (np.maximum(((data["visitNumber"])), (((9.72239112854003906))))))))) +
            0.100000*np.tanh(((((((((((((data["t_sum_log"]) - (np.maximum((((6.0))), ((data["visitNumber"])))))) * 2.0)) * 2.0)) - (-2.0))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(np.minimum((((((data["t_median"]) < ((7.85152149200439453)))*1.))), ((((((data["next_session_2"]) + (((((data["pred_0"]) - ((7.85152149200439453)))) * 2.0)))) * 2.0))))) +
            0.100000*np.tanh(((((((((((data["t_sum_log"]) - (np.maximum(((data["visitNumber"])), ((((3.0) * 2.0))))))) * 2.0)) * 2.0)) + (3.0))) * 2.0)) +
            0.100000*np.tanh((((14.69243335723876953)) * (((((data["pred_3"]) - (((data["pred_0"]) + (((data["geoNetwork_country"]) + ((10.17979812622070312)))))))) + (data["pred_3"]))))) +
            0.100000*np.tanh(((((((((((((data["t_sum_act"]) - ((((10.99480342864990234)) / 2.0)))) * 2.0)) * ((10.99802207946777344)))) + (data["t_sum_log"]))) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((((((10.0)) * (((data["pred_3"]) - ((6.0)))))) + (3.0))) * ((12.68457317352294922)))) +
            0.100000*np.tanh((((((6.0)) - (((data["pred_0"]) - (data["t_median"]))))) * (((((data["pred_0"]) - ((7.13541793823242188)))) * ((8.68632888793945312)))))) +
            0.100000*np.tanh(((((((np.minimum(((((data["t_sum_log"]) - ((5.0))))), ((((data["next_session_2"]) - (data["visitNumber"])))))) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((data["t_sum_act"]) * 2.0)) - (((data["pred_0"]) - (data["pred_3"]))))) - ((10.0)))) * ((10.0)))) +
            0.100000*np.tanh((((13.06485366821289062)) * ((((((((13.06485366821289062)) * (((((data["t_sum_log"]) - (-3.0))) - ((8.01604843139648438)))))) * 2.0)) * 2.0)))) +
            0.100000*np.tanh(((((((data["t_sum_log"]) - (np.maximum(((data["t_median"])), ((np.maximum(((data["visitNumber"])), (((((10.0)) - (data["t_sum_log"]))))))))))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((np.where(data["pred_3"]<0, ((data["t_sum_log"]) + (data["t_sum_log"])), data["t_sum_log"] )) - ((10.0)))) * ((5.26560354232788086)))) +
            0.100000*np.tanh(((((((data["t_mean"]) - (data["t_sum_log"]))) + (((((data["t_sum_log"]) - ((10.0)))) * ((11.74072933197021484)))))) * 2.0)) +
            0.100000*np.tanh(((((((((data["t_sum_act"]) * 2.0)) - ((10.21498489379882812)))) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((((((10.75512409210205078)) * (((((data["t_sum_log"]) - (np.tanh((data["pred_2"]))))) - ((10.75512409210205078)))))) + ((8.73960781097412109)))) * 2.0)) +
            0.100000*np.tanh((((14.71356868743896484)) * ((((((14.71356868743896484)) * (((data["pred_0"]) + (((-3.0) * 2.0)))))) + (((-3.0) * 2.0)))))) +
            0.100000*np.tanh((((9.81327152252197266)) * (((((((((7.16616725921630859)) > (data["t_mean"]))*1.)) * (data["t_sum_act"]))) + (((data["t_sum_act"]) - ((9.76804924011230469)))))))) +
            0.100000*np.tanh(((((data["t_sum_act"]) + (((data["t_sum_log"]) - (np.maximum(((((((data["totals_hits"]) * 2.0)) + (data["t_sum_log"])))), (((9.55896186828613281))))))))) * 2.0)) +
            0.100000*np.tanh(((((((((((((data["t_sum_log"]) * 2.0)) - ((10.0)))) - (data["geoNetwork_country"]))) * 2.0)) * 2.0)) - (data["visitNumber"]))) +
            0.100000*np.tanh(((((data["t_sum_log"]) - (np.where(((data["pred_1"]) - (2.0))<0, (11.79721641540527344), ((data["visitNumber"]) + (data["t_median"])) )))) * 2.0)) +
            0.100000*np.tanh((((((((((11.64218807220458984)) * (((data["pred_3"]) - ((5.93939924240112305)))))) * 2.0)) - (-3.0))) * ((5.93939924240112305)))) +
            0.100000*np.tanh((((4.71989488601684570)) * ((((((((4.71989488601684570)) * (((((data["t_sum_log"]) - ((4.71989488601684570)))) * 2.0)))) + (data["pred_0"]))) * 2.0)))) +
            0.100000*np.tanh(((((((np.where(((data["t_sum_act"]) - ((7.0)))<0, ((data["t_sum_log"]) * 2.0), data["t_sum_act"] )) - ((9.0)))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((data["t_sum_log"]) - ((10.06210994720458984)))) * ((13.34921360015869141)))) - (data["t_sum_log"]))) - ((10.0)))) +
            0.100000*np.tanh(((((((((data["t_sum_log"]) * 2.0)) - ((8.0)))) * ((((7.0)) - (data["t_sum_log"]))))) * ((8.0)))) +
            0.100000*np.tanh(np.minimum(((((((((np.minimum(((((((data["t_mean"]) - ((5.0)))) * 2.0))), ((data["trafficSource_source"])))) * 2.0)) * 2.0)) * 2.0))), ((data["geoNetwork_metro"])))) +
            0.100000*np.tanh(np.minimum((((((((8.0)) * (((((data["t_sum_log"]) * 2.0)) - ((8.0)))))) * 2.0))), (((((7.0)) - (data["t_sum_log"])))))) +
            0.100000*np.tanh(((((((-3.0) + (np.where(data["pred_2"] < -99998, np.where(data["pred_1"] < -99998, -3.0, data["pred_0"] ), data["t_mean"] )))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((data["pred_0"]) - (2.0))) - (np.where(data["pred_1"]<0, data["totals_hits"], data["pred_0"] )))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((((((data["pred_0"]) * 2.0)) + (np.minimum(((data["pred_1"])), ((data["pred_1"])))))) - ((8.0)))) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((-2.0) * 2.0)) + (data["t_sum_log"]))) * ((((6.0)) - (data["t_sum_log"]))))) * ((6.22247934341430664)))) +
            0.100000*np.tanh(((((((data["t_sum_log"]) - (np.maximum(((np.maximum(((data["pred_0"])), ((((((1.0) * 2.0)) * 2.0)))))), ((data["visitNumber"])))))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((data["t_sum_log"]) + (((np.where(data["pred_2"]>0, -3.0, ((data["t_sum_act"]) - ((6.0))) )) * 2.0)))) * 2.0)) * 2.0)) +
            0.100000*np.tanh((((14.27134418487548828)) * ((((((11.12297344207763672)) * (((((data["geoNetwork_continent"]) - ((9.0)))) + (data["t_sum_log"]))))) + (data["next_session_1"]))))) +
            0.100000*np.tanh(np.where(((data["pred_0"]) + (((-2.0) * 2.0)))>0, data["pred_0"], ((np.minimum(((-3.0)), ((data["pred_3"])))) + (data["t_mean"])) )) +
            0.100000*np.tanh(((((((np.where(((-3.0) + (data["totals_hits"]))<0, data["t_sum_act"], 2.0 )) - ((4.0)))) * 2.0)) * 2.0)) +
            0.100000*np.tanh(np.minimum((((((4.68725061416625977)) - (((data["totals_hits"]) - (-2.0)))))), ((np.minimum(((data["t_median"])), ((((data["t_sum_log"]) - ((4.68725061416625977)))))))))) +
            0.100000*np.tanh(np.minimum(((((data["pred_0"]) * ((12.75279808044433594))))), ((((((data["t_sum_act"]) * (np.maximum(((data["t_sum_log"])), ((data["pred_0"])))))) - ((12.75279808044433594))))))) +
            0.100000*np.tanh((((13.62804603576660156)) * (((((((((4.70133781433105469)) - ((9.57412528991699219)))) + (((((data["pred_1"]) * 2.0)) * 2.0)))/2.0)) - (data["t_median"]))))) +
            0.100000*np.tanh(((((((np.minimum(((((((((data["t_sum_log"]) + (-3.0))) * 2.0)) * 2.0))), ((data["totals_pageviews"])))) * 2.0)) + (-3.0))) * 2.0)) +
            0.100000*np.tanh(np.minimum((((((5.0)) * (((data["t_sum_act"]) - ((((6.49275684356689453)) - (data["t_sum_act"])))))))), (((((5.0)) - (data["pred_0"])))))) +
            0.100000*np.tanh((((((9.0)) * (((((data["t_median"]) - ((9.0)))) - (data["visitNumber"]))))) + (((data["t_sum_log"]) / 2.0)))) +
            0.100000*np.tanh(((((((((((((data["t_sum_log"]) + (-3.0))) * 2.0)) + (((data["pred_0"]) - (data["totals_hits"]))))) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((data["t_median"]) + (((((((((((np.maximum(((data["t_median"])), ((data["t_sum_log"])))) - ((10.68563747406005859)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +
            0.100000*np.tanh(((((((((-3.0) + (np.where(data["pred_12"] < -99998, np.maximum(((data["t_mean"])), ((data["pred_0"]))), data["pred_12"] )))) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((((np.minimum(((data["pred_1"])), ((((np.minimum(((2.0)), ((data["pred_2"])))) - (data["pred_1"])))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((np.minimum(((((np.where(data["geoNetwork_country"]<0, data["t_sum_log"], -3.0 )) - (3.0)))), ((((data["next_session_1"]) - (data["visitNumber"])))))) * 2.0)) +
            0.100000*np.tanh(((((np.where(((-2.0) + (data["pred_1"]))>0, -3.0, ((-3.0) + (data["t_sum_log"])) )) * 2.0)) * 2.0)) +
            0.100000*np.tanh(np.minimum((((((5.0)) - (data["t_mean"])))), ((((((data["t_sum_log"]) - ((((5.0)) - (data["t_sum_log"]))))) * 2.0))))) +
            0.100000*np.tanh(((-2.0) + (((((((((data["pred_0"]) + (-3.0))) * 2.0)) * 2.0)) + (((data["t_sum_log"]) * (data["next_session_1"]))))))) +
            0.100000*np.tanh(((((((data["t_sum_log"]) - (np.maximum(((2.0)), ((((((data["t_sum_log"]) - (2.0))) * 2.0))))))) * ((12.37948131561279297)))) * 2.0)) +
            0.100000*np.tanh(((data["t_sum_act"]) - (np.maximum((((((4.0)) * ((((4.94181728363037109)) * ((((data["next_session_1"]) > (data["next_session_2"]))*1.))))))), (((4.0))))))) +
            0.100000*np.tanh(((((np.where(data["geoNetwork_continent"]>0, -3.0, np.where(data["pred_2"]>0, data["pred_2"], ((data["t_sum_log"]) * 2.0) ) )) * 2.0)) - ((6.66043043136596680)))) +
            0.100000*np.tanh(((((data["t_sum_log"]) - (np.maximum(((np.where(data["channelGrouping"]<0, data["totals_pageviews"], data["t_sum_log"] ))), ((np.maximum(((data["pred_0"])), ((3.0))))))))) * 2.0)) +
            0.100000*np.tanh(((((((((np.minimum(((((data["t_sum_act"]) + (-2.0)))), ((((data["geoNetwork_country"]) * (data["pred_1"])))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            0.100000*np.tanh(((((((data["pred_13"]) * 2.0)) * ((((2.0) > (np.where(data["pred_13"] < -99998, data["t_sum_log"], data["pred_12"] )))*1.)))) * ((14.28744792938232422)))) +
            0.100000*np.tanh(np.where(data["pred_1"]>0, ((np.where(data["pred_0"]>0, (6.0), data["pred_1"] )) - (data["t_median"])), ((data["t_median"]) - ((6.0))) )) +
            0.100000*np.tanh(((((((((np.minimum(((data["t_sum_log"])), ((data["totals_pageviews"])))) * 2.0)) + ((((-3.0) + (-2.0))/2.0)))) * ((7.0)))) * 2.0)) +
            0.100000*np.tanh(((np.where(data["pred_13"] < -99998, data["t_mean"], data["pred_13"] )) + (((((np.maximum(((data["pred_12"])), ((data["pred_13"])))) * 2.0)) + (-3.0))))) +
            0.100000*np.tanh(np.where(data["pred_29"] < -99998, ((((((((-1.0) + (((data["t_sum_log"]) * 2.0)))) * 2.0)) * 2.0)) + (-3.0)), -3.0 )) +
            0.100000*np.tanh(((((((((data["pred_13"]) * 2.0)) * 2.0)) * 2.0)) * ((((data["pred_200"]) < (((data["t_sum_act"]) * (((data["pred_13"]) * 2.0)))))*1.)))) +
            0.099658*np.tanh(((np.tanh((np.tanh((np.where((((((-3.0) / 2.0)) + (data["pred_12"]))/2.0)>0, (3.79615640640258789), ((-3.0) / 2.0) )))))) * 2.0)) +
            0.100000*np.tanh(((data["pred_254"]) * (np.maximum(((((data["pred_254"]) - (-3.0)))), (((((data["pred_29"]) > (((data["pred_148"]) - (data["pred_41"]))))*1.))))))) +
            0.099951*np.tanh(((data["pred_132"]) - (np.where(data["pred_29"]<0, ((data["pred_41"]) - ((((((data["pred_132"]) < (data["pred_29"]))*1.)) * 2.0))), (4.58089542388916016) )))) +
            0.099756*np.tanh(np.where(data["pred_12"] < -99998, 0.0, np.where(data["pred_43"] < -99998, data["pred_12"], data["pred_254"] ) )) +
            0.099707*np.tanh(np.where(data["pred_29"]>0, data["pred_148"], ((np.minimum(((data["pred_29"])), ((((((data["pred_132"]) - (data["pred_41"]))) * 2.0))))) - (data["pred_254"])) )) +
            0.099316*np.tanh(((np.where(np.where(data["pred_132"] < -99998, data["pred_29"], data["pred_254"] ) < -99998, data["pred_29"], ((data["pred_254"]) - (data["pred_41"])) )) - (data["pred_29"]))) +
            0.099951*np.tanh(np.where(data["pred_29"]>0, -3.0, np.where(data["pred_132"]>0, -3.0, ((data["pred_132"]) - (np.minimum(((-3.0)), ((data["pred_29"]))))) ) )) +
            0.100000*np.tanh(((((((np.where(data["pred_11"]<0, data["pred_254"], ((data["pred_11"]) - (data["pred_29"])) )) - (data["pred_29"]))) * 2.0)) * 2.0)) +
            0.099609*np.tanh(((((data["pred_29"]) - (((data["pred_148"]) - (np.maximum(((((data["pred_254"]) - (data["pred_29"])))), ((data["pred_198"])))))))) * (data["pred_254"]))) +
            0.099805*np.tanh((((((np.where(data["pred_132"] < -99998, data["pred_41"], 3.0 )) + (data["pred_132"]))/2.0)) - (np.where(data["pred_29"]<0, data["pred_41"], (6.78871536254882812) )))) +
            0.099853*np.tanh(((((np.minimum((((((data["pred_29"]) > (data["pred_88"]))*1.))), ((((((data["pred_111"]) - (data["pred_41"]))) - (data["pred_29"])))))) * 2.0)) * 2.0)) +
            0.099560*np.tanh(np.where(data["pred_29"]>0, -2.0, np.where(data["pred_41"]>0, -2.0, ((data["pred_153"]) - (np.minimum(((data["pred_41"])), ((-3.0))))) ) )) +
            0.099707*np.tanh(((np.where(data["pred_29"]>0, data["pred_251"], (((data["pred_29"]) > (data["pred_136"]))*1.) )) + (((data["pred_132"]) - (data["pred_41"]))))) +
            0.099463*np.tanh(((((data["pred_29"]) - (data["pred_136"]))) * ((((11.27190399169921875)) + (((data["pred_111"]) + ((12.76537227630615234)))))))) +
            0.100000*np.tanh(np.maximum(((((data["pred_132"]) - ((((data["pred_251"]) + (data["pred_13"]))/2.0))))), ((((((((((data["pred_13"]) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) +
            0.099805*np.tanh(np.where(data["pred_29"]>0, data["pred_251"], ((data["pred_132"]) - (np.where(data["pred_148"] < -99998, data["pred_43"], ((((data["pred_251"]) * 2.0)) * 2.0) ))) )) +
            0.099658*np.tanh(np.where(data["pred_29"]>0, data["pred_254"], np.where(data["pred_153"] < -99998, ((data["pred_29"]) - (data["pred_43"])), ((data["pred_29"]) - (data["pred_153"])) ) )) +
            0.099805*np.tanh(np.maximum(((((((data["pred_254"]) * 2.0)) * 2.0))), ((((np.minimum(((data["pred_41"])), ((((((data["pred_132"]) / 2.0)) / 2.0))))) - (data["pred_41"])))))) +
            0.099756*np.tanh(np.where(data["pred_29"]>0, data["pred_254"], ((np.where(data["pred_41"] < -99998, data["pred_29"], ((data["pred_148"]) - (-1.0)) )) - (data["pred_41"])) )) +
            0.099707*np.tanh(((data["pred_148"]) - (np.where(data["pred_29"]>0, (7.0), np.where(data["pred_148"]>0, (7.0), data["pred_41"] ) )))) +
            0.099805*np.tanh(np.where(np.maximum(((data["pred_111"])), ((data["pred_29"])))>0, -3.0, ((data["pred_111"]) - (np.minimum(((data["pred_41"])), ((-3.0))))) )) +
            0.100000*np.tanh(np.where(data["pred_61"]>0, (9.0), ((data["pred_111"]) - (data["pred_29"])) )) +
            0.099170*np.tanh(np.where(data["pred_29"]<0, ((np.where(data["pred_41"] < -99998, data["pred_29"], data["pred_132"] )) - ((((data["pred_41"]) + (data["pred_251"]))/2.0))), data["pred_254"] )) +
            0.099609*np.tanh(((((data["pred_29"]) - (data["pred_153"]))) * (((((((data["pred_29"]) + (data["pred_41"]))) - (data["pred_132"]))) * (data["pred_61"]))))) +
            0.099609*np.tanh(np.where(data["pred_153"]>0, -3.0, np.where(data["pred_29"]>0, -3.0, ((data["pred_153"]) - (np.minimum(((-3.0)), ((data["pred_43"]))))) ) )) +
            0.099609*np.tanh(((data["pred_29"]) - (np.where(np.where(data["pred_29"]<0, data["pred_41"], data["pred_29"] ) < -99998, data["pred_61"], ((data["pred_61"]) * (data["pred_200"])) )))))


maxval = np.log1p(trn_user_target['target']).max()
maxval
print(mean_squared_error(np.log1p(trn_user_target['target']), GP1(gp_trn_users).clip(0,maxval)) ** .5)

a = GP1(gp_sub_users).clip(0,maxval).values
sub_full_data['PredictedLogRevenue'] = a
sub_full_data[['PredictedLogRevenue']].to_csv('gp1.csv', index=True)

sub_full_data.PredictedLogRevenue.min()
sub_full_data.PredictedLogRevenue.max()