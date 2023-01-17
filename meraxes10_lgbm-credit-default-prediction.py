import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import random
import gc
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
gc.enable()
train_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
test_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
train_counts = train_data.count().sort_values()/len(train_data)
test_counts = test_data.count().sort_values()/len(test_data)
cols = set(train_counts[(train_counts < 1) & (train_counts > 0.99)].index) - set(test_counts[(test_counts < 1) & (test_counts > 0.9)].index)
cols
train_data.dropna(subset=cols, inplace=True)
train_target = train_data[['SK_ID_CURR', 'TARGET']]
submit = test_data[['SK_ID_CURR']]
train_data.drop(columns=['TARGET'], inplace=True)
test_data['IS_TRAIN'] = 0
train_data['IS_TRAIN'] = 1
application_data = train_data.append(test_data)
del(train_data)
del(test_data)
appl_counts = application_data.count().sort_values()/len(application_data)
appl_counts[(appl_counts < 0.6)]
appl_counts = application_data.count().sort_values()/len(application_data)
cols = list(set(appl_counts[(appl_counts < 0.6)].index) - set(['EXT_SOURCE_1', 'OWN_CAR_AGE']))
#application_data.drop(columns=cols, inplace=True)
le = LabelEncoder()
for col in application_data.select_dtypes('object'):
    if len(application_data[col].unique()) <= 2:
        le.fit(application_data[col])
        application_data[col] = le.transform(application_data[col])
application_data = pd.get_dummies(application_data, dummy_na=True)
appl_counts = application_data.count().sort_values()/len(application_data)
appl_counts[(appl_counts < 1)]
train_data = application_data[application_data.IS_TRAIN == 1].merge(train_target, how='left', on='SK_ID_CURR')
corrs = train_data.corr()
del(train_data)
corrs['TARGET'].abs().sort_values().tail(40)
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
application_data[application_data.AMT_REQ_CREDIT_BUREAU_WEEK.isnull() & application_data.SK_ID_CURR.isin(bureau.SK_ID_CURR.unique())]
application_data['IS_IN_BUREAU'] = 0
application_data.loc[application_data.SK_ID_CURR.isin(bureau.SK_ID_CURR.unique()), 'IS_IN_BUREAU'] = 1
appl_counts = application_data.count().sort_values()/len(application_data)
appl_counts[(appl_counts < 1)]
application_data['HAS_SOCIAL_CIRCLE'] = 0
application_data.loc[~application_data.OBS_30_CNT_SOCIAL_CIRCLE.isnull(), 'HAS_SOCIAL_CIRCLE'] = 1
application_data
application_data['AMT_CREDIT_FRAC'] = application_data.AMT_CREDIT / application_data.AMT_INCOME_TOTAL
application_data['AMT_CREDIT_FRAC'] = application_data.AMT_ANNUITY / application_data.AMT_CREDIT
application_data['AMT_GOODS_FRAC'] = application_data.AMT_GOODS_PRICE / application_data.AMT_CREDIT
application_data['AMT_ANNUITY_FRAC'] = application_data.AMT_ANNUITY / application_data.AMT_INCOME_TOTAL
application_data['AMT_DPD_DEF'] = application_data.DEF_30_CNT_SOCIAL_CIRCLE + application_data.OBS_30_CNT_SOCIAL_CIRCLE
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
bureau = bureau[bureau.SK_ID_CURR.isin(application_data.SK_ID_CURR.unique())]
bureau_balance = bureau_balance[bureau_balance.SK_ID_BUREAU.isin(bureau.SK_ID_BUREAU.unique())]
bureau_balance = pd.get_dummies(bureau_balance)
bureau_balance = bureau_balance.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'])
temp = bureau_balance.groupby('SK_ID_BUREAU').size().to_frame()
temp = temp.rename(columns={0: 'COUNT'})
temp.reset_index(inplace=True)
bureau_balance = bureau_balance.groupby('SK_ID_BUREAU').agg({'last', 'sum', 'mean'})
bureau_balance.columns = bureau_balance.columns.map('_'.join)
bureau_balance.reset_index(inplace=True)
bureau_balance = bureau_balance.merge(temp, how='left', on='SK_ID_BUREAU')
bureau_balance.columns = bureau_balance.columns.map(lambda x : 'BLN_' + x if x != 'SK_ID_BUREAU' else x)
bureau = bureau.merge(bureau_balance, how='left', on='SK_ID_BUREAU')
bureau.drop(columns='SK_ID_BUREAU', inplace=True)
del(bureau_balance)
bureau = bureau.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'])
bureau = pd.get_dummies(bureau, dummy_na=True)
temp = bureau.groupby('SK_ID_CURR').size().to_frame()
temp = temp.rename(columns={0: 'COUNT'})
temp.reset_index(inplace=True)
bureau = bureau.groupby('SK_ID_CURR').agg({'sum', 'mean', 'max'})
bureau.columns = bureau.columns.map('_'.join)
bureau.reset_index(inplace=True)
bureau = bureau.merge(temp, how='left', on='SK_ID_CURR')
bureau.columns = bureau.columns.map(lambda x : 'BRU_' + x if x != 'SK_ID_CURR' else x)
application_data = application_data.merge(bureau, how='left', on='SK_ID_CURR')
del(bureau)
prev_application = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
prev_application = prev_application[prev_application.SK_ID_CURR.isin(application_data.SK_ID_CURR.unique())]
prev_application = pd.get_dummies(prev_application, dummy_na=True)
prev_application.drop(columns='SK_ID_PREV', inplace=True)
prev_application = prev_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
temp = prev_application.groupby('SK_ID_CURR').size().to_frame()
temp = temp.rename(columns={0: 'COUNT'})
temp.reset_index(inplace=True)
prev_application = prev_application.groupby('SK_ID_CURR').agg(['max', 'sum', 'mean']) # last
prev_application.columns = prev_application.columns.map('_'.join)
prev_application.reset_index(inplace=True)
prev_application = prev_application.merge(temp, how='left', on='SK_ID_CURR')
prev_application.columns = prev_application.columns.map(lambda x : 'PREV_' + x if x != 'SK_ID_CURR' else x)
application_data = application_data.merge(prev_application, how='left', on='SK_ID_CURR')
del(prev_application)
pos_cash_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
pos_cash_balance = pos_cash_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
temp = pos_cash_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().to_frame()
temp = temp.rename(columns={0: 'BLN_COUNT'})
temp.reset_index(inplace=True)
pos_cash_balance = pd.get_dummies(pos_cash_balance, dummy_na=True)
pos_cash_balance = pos_cash_balance.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg(['sum', 'mean', 'max']) # last
pos_cash_balance.columns = pos_cash_balance.columns.map('_'.join)
pos_cash_balance.reset_index(inplace=True)
pos_cash_balance = pos_cash_balance.merge(temp, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])
pos_cash_balance.drop(columns='SK_ID_PREV', inplace=True)
temp = pos_cash_balance.groupby('SK_ID_CURR').size().to_frame()
temp = temp.rename(columns={0: 'COUNT'})
temp.reset_index(inplace=True)
pos_cash_balance = pos_cash_balance.groupby(['SK_ID_CURR']).agg(['sum', 'mean', 'max']) 
pos_cash_balance.columns = pos_cash_balance.columns.map('_'.join)
pos_cash_balance.reset_index(inplace=True)
pos_cash_balance = pos_cash_balance.merge(temp, how='left', on='SK_ID_CURR')
pos_cash_balance.columns = pos_cash_balance.columns.map(lambda x : 'CSH_' + x if x != 'SK_ID_CURR' else x)
application_data = application_data.merge(pos_cash_balance, how='left', on='SK_ID_CURR')
del(pos_cash_balance)
credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
credit_card_balance = pd.get_dummies(credit_card_balance, dummy_na=True)
credit_card_balance = credit_card_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
temp = credit_card_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().to_frame()
temp = temp.rename(columns={0: 'COUNT'})
temp.reset_index(inplace=True)
credit_card_balance = credit_card_balance.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg(['sum', 'mean', 'max']) # last
credit_card_balance.columns = credit_card_balance.columns.map('_'.join)
credit_card_balance.reset_index(inplace=True)
credit_card_balance = credit_card_balance.merge(temp, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])
credit_card_balance.drop(columns='SK_ID_PREV', inplace=True)
credit_card_balance = credit_card_balance.groupby(['SK_ID_CURR']).agg(['sum', 'mean'])
credit_card_balance.columns = credit_card_balance.columns.map('_'.join)
credit_card_balance.reset_index(inplace=True)
credit_card_balance.columns = credit_card_balance.columns.map(lambda x : 'CRD_' + x if x != 'SK_ID_CURR' else x)
application_data = application_data.merge(credit_card_balance, how='left', on='SK_ID_CURR')
del(credit_card_balance)
installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv', 
                               na_values=['XNA', 'XAP'], na_filter=True)
installments_payments = installments_payments.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT'])
temp = installments_payments.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().to_frame()
temp = temp.rename(columns={0: 'COUNT'})
temp.reset_index(inplace=True)
installments_payments.fillna(0, inplace=True)
installments_payments = installments_payments.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg(['sum', 'mean', 'max', 'min'])
installments_payments.columns = installments_payments.columns.map('_'.join)
installments_payments.reset_index(inplace=True)
installments_payments = installments_payments.merge(temp, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])
installments_payments.drop(columns='SK_ID_PREV', inplace=True)
installments_payments = installments_payments.groupby(['SK_ID_CURR']).agg(['sum', 'mean', 'max', 'min'])
installments_payments.columns = installments_payments.columns.map('_'.join)
installments_payments.reset_index(inplace=True)
installments_payments.columns = installments_payments.columns.map(lambda x : 'INS_' + x if x != 'SK_ID_CURR' else x)
application_data = application_data.merge(installments_payments, how='left', on='SK_ID_CURR')
del(installments_payments)
for col in application_data.columns:
    if len(application_data[col].unique()) <= 1:
        application_data.drop(columns=col,inplace=True)
application_data.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in application_data.columns]
model = lgb.LGBMClassifier()
train_data = application_data[application_data.IS_TRAIN == 1.0]
test_data = application_data[application_data.IS_TRAIN == 0.0]
train_data.drop(columns='IS_TRAIN', inplace=True)
test_data.drop(columns='IS_TRAIN', inplace=True)
del(application_data)
params = model.get_params()
params['objective'] = 'binary'
params['metric'] = 'auc'
skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
final_importance = np.zeros(len(train_data.columns))
for n_fold, (train_index, valid_index) in tqdm(enumerate(skf.split(train_data, train_target.TARGET))):
    X_train = train_data.iloc[train_index]
    y_train = train_target.iloc[train_index].TARGET
    X_valid = train_data.iloc[valid_index]
    y_valid = train_target.iloc[valid_index].TARGET
    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_eval = lgb.Dataset(data=X_valid, label=y_valid)
    model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=100)
    final_importance += model.feature_importance()
fi = pd.DataFrame()
fi['FEAT'] = train_data.columns
fi['importance'] = final_importance
fi = fi.sort_values(by='importance', ascending=False)
fi = fi[fi.importance != 0]
fi.head(30)
cols = list(set(fi.FEAT.values).union(set(['SK_ID_CURR'])))
len(cols)
train_data = train_data[cols]
test_data = test_data[cols]
def get_random_params():
    params = {
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'num_leaves': random.randint(10, 60),
        'max_depth': random.randint(10, 30),
        'learning_rate': random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01]),
        'n_estimators': random.randint(1000, 20000),
        'objective': 'binary',
        'reg_alpha': random.choice([0.001, 0.005, 0.01, 0.05, 0.1]),
        'reg_lambda': random.choice([0.001, 0.005, 0.01, 0.05, 0.1]),       
        'colsample_bytree': random.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'min_child_samples': random.randint(10, 100),
        'subsample_for_bin': random.randint(50000, 300000)
    }
    return params
best_params = {'boosting_type': 'gbdt', 
               'metric': 'auc', 
               'num_leaves': 46, 
               'max_depth': 18, 
               'learning_rate': 0.01, 
               'n_estimators': 6289, 
               'objective': 'binary', 
               'reg_alpha': 0.05, 
               'reg_lambda': 0.05, 
               'colsample_bytree': 0.4, 
               'min_child_samples': 79, 
               'subsample_for_bin': 113092}
best_auc = 0.787228
def get_best_params(hyper_rounds, n_folds, best_params=None, best_auc=0):
    best_params = best_params
    best_auc = best_auc
    lgb_train = lgb.Dataset(data=train_data, label=train_target.TARGET)
    for i in tqdm(range(hyper_rounds)):
        curr_params = get_random_params()
        start = time.time()
        print(curr_params)
        eval_hist = lgb.cv(curr_params, lgb_train, early_stopping_rounds = 200, nfold = n_folds, seed = 42, verbose_eval = 100)
        end = time.time()
        print('TIME:', end-start)
        curr_auc = eval_hist['auc-mean'][-1]
        if curr_auc > best_auc:
            best_params = curr_params
            best_auc = curr_auc
    return best_params, best_auc
HYPER_ROUNDS = 1
FOLDS = 5
#best_params, best_auc = get_best_params(HYPER_ROUNDS, FOLDS, best_params, best_auc)
N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
sub_preds = np.zeros(len(test_data))
avg_valid_auc = 0
for n_fold, (train_index, valid_index) in tqdm(enumerate(skf.split(train_data, train_target.TARGET))):
    print("FOLD N:", n_fold)
    X_train = train_data.iloc[train_index]
    y_train = train_target.iloc[train_index].TARGET
    X_valid = train_data.iloc[valid_index]
    y_valid = train_target.iloc[valid_index].TARGET
    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_eval = lgb.Dataset(data=X_valid, label=y_valid)
    model = lgb.train(best_params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=100)
    y_pred = model.predict(X_valid)
    sub_preds += model.predict(test_data) / skf.n_splits
    avg_valid_auc += roc_auc_score(y_valid, y_pred) / N_FOLDS
avg_valid_auc
submit['TARGET'] = sub_preds
submit.to_csv('submission.csv', index = False)