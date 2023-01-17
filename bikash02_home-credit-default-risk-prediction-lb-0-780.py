import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
full_train_orig = pd.read_csv("../Data/application_train.csv")
test =  pd.read_csv("../Data/application_test.csv")
### Load bureau data
bureau = pd.read_csv("../Data/bureau.csv")
bureau_bal = pd.read_csv("../Data/bureau_balance.csv")

bureau_bal_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
# for col in bb_cat:
#     bb_aggregations[col] = ['mean']
bb_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(bureau_bal_aggregations)
bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
#bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
#bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

bureau_gb = bureau.groupby("SK_ID_CURR").agg({'DAYS_CREDIT':['mean', 'max', 'min'], 'CREDIT_DAY_OVERDUE':['mean', 'max', 'min'],
                                             'CNT_CREDIT_PROLONG': ['sum', 'max'], 'AMT_CREDIT_SUM':'mean',
                                             'DAYS_CREDIT_UPDATE':['max', 'min'], 'AMT_CREDIT_SUM_OVERDUE':['mean'],
                                             'SK_ID_BUREAU':'count'})
# bureau_gb = bureau.groupby("SK_ID_CURR")[["DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "CNT_CREDIT_PROLONG", "DAYS_CREDIT_UPDATE"
#                              , "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM"]].mean().add_suffix("_bur").reset_index()
bureau_gb.columns = ['_bureau_'.join(col) for col in bureau_gb.columns]
bureau_gb = bureau_gb.reset_index()

### OHE categorical features and combine
bureau_cats = pd.get_dummies(bureau.select_dtypes('object').drop("CREDIT_CURRENCY", axis=1))
bureau_cats['SK_ID_CURR'] = bureau['SK_ID_CURR']
bureau_cats_grouped = bureau_cats.groupby('SK_ID_CURR').agg('sum').reset_index()
bureau_gb = pd.merge(bureau_gb, bureau_cats_grouped, on = 'SK_ID_CURR', how = 'left')
### Load bureau data


# bureau_gb = bureau.groupby("SK_ID_CURR").agg({'DAYS_CREDIT':['mean', 'max', 'min'], 'CREDIT_DAY_OVERDUE':['mean', 'max', 'min'],
#                                              'CNT_CREDIT_PROLONG': ['sum', 'max'], 'AMT_CREDIT_SUM':'mean',
#                                              'DAYS_CREDIT_UPDATE':['max', 'min'], 'AMT_CREDIT_SUM_OVERDUE':['mean']})
# # bureau_gb = bureau.groupby("SK_ID_CURR")[["DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "CNT_CREDIT_PROLONG", "DAYS_CREDIT_UPDATE"
# #                              , "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM"]].mean().add_suffix("_bur").reset_index()
# bureau_gb.columns = ['_'.join(col) for col in bureau_gb.columns]
# bureau_gb = bureau_gb.reset_index()

# ### OHE categorical features and combine
# bureau_cats = pd.get_dummies(bureau.select_dtypes('object'))
# bureau_cats['SK_ID_CURR'] = bureau['SK_ID_CURR']
# bureau_cats_grouped = bureau_cats.groupby('SK_ID_CURR').agg('sum').reset_index()
# bureau_gb = pd.merge(bureau_gb, bureau_cats_grouped, on = 'SK_ID_CURR', how = 'left')
### Load previous application data
prv = pd.read_csv("../Data/previous_application.csv")
prv['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
prv['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
prv['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
prv['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
prv['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
prv['APP_CREDIT_PERC'] = prv['AMT_APPLICATION'] / prv['AMT_CREDIT']
prv_gb = prv.groupby("SK_ID_CURR").agg({"AMT_ANNUITY":['mean'], "AMT_APPLICATION":['mean'], "AMT_CREDIT":['mean'],
                                        "AMT_DOWN_PAYMENT":['mean'], "AMT_GOODS_PRICE":['mean'], "SELLERPLACE_AREA":['mean'],
                                        "DAYS_DECISION":['min', 'max'], "DAYS_TERMINATION":['min', 'max'], "DAYS_LAST_DUE":['min', 'max'],
                                        "DAYS_FIRST_DUE":['min', 'max'], "DAYS_LAST_DUE_1ST_VERSION":['min', 'max'],
                                        "SK_ID_PREV":['count']})
prv_gb.columns = ['_prev_'.join(col) for col in prv_gb.columns]
prv_gb = prv_gb.reset_index()

prv_cats = pd.get_dummies(prv.select_dtypes('object').drop(["NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "NAME_CONTRACT_TYPE"], axis=1))
prv_cats['SK_ID_CURR'] = prv['SK_ID_CURR']
prv_cats_grouped = prv_cats.groupby('SK_ID_CURR').agg('sum').reset_index()
prv_gb = pd.merge(prv_gb, prv_cats_grouped, on = 'SK_ID_CURR', how = 'left')
prv_gb.head(2)
### installment data 
inst = pd.read_csv("../Data/installments_payments.csv")
inst["days_diff"] = inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]
inst["amt_diff"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
inst['PAYMENT_PERC'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']
# Days past due and days before due (no negative values)
inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['DBD'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']
inst['DPD'] = inst['DPD'].apply(lambda x: x if x > 0 else 0)
inst['DBD'] = inst['DBD'].apply(lambda x: x if x > 0 else 0)
    
inst_gb = inst.groupby("SK_ID_CURR").agg({"days_diff":['mean', 'max', 'min'], "amt_diff":['mean', 'max', 'min'],
                                        "NUM_INSTALMENT_VERSION":[ 'max', 'count'], "NUM_INSTALMENT_NUMBER":['max', 'count'], 
                                          "SK_ID_PREV":['count']})
inst_gb.columns = ['_inst_'.join(col) for col in inst_gb.columns]
inst_gb = inst_gb.reset_index()
# inst_gb = inst.groupby("SK_ID_CURR")[["AMT_INSTALMENT","AMT_PAYMENT"]].sum().add_suffix("_inst").reset_index()
# inst_gb["diff"] = inst_gb["AMT_INSTALMENT_inst"] - inst_gb["AMT_PAYMENT_inst"]
# inst_gb = inst_gb.drop(["AMT_INSTALMENT_inst", "AMT_PAYMENT_inst"], axis=1)
inst_gb.head(2)
### POS_Cash data
cash = pd.read_csv("../Data/POS_CASH_balance.csv")
cash_gb = cash.groupby("SK_ID_CURR").agg({"SK_ID_PREV":['count']})
cash_gb.columns = ['_cash_'.join(col) for col in cash_gb.columns]
cash_gb = cash_gb.reset_index()

# cash_cats = pd.get_dummies(cash.select_dtypes('object'))
# cash_cats['SK_ID_CURR'] = cash['SK_ID_CURR']
# cash_cats_grouped = cash_cats.groupby('SK_ID_CURR').agg('sum').reset_index()
# cash_gb = pd.merge(cash_gb, cash_cats_grouped, on = 'SK_ID_CURR', how = 'left')

# cash_cats = pd.get_dummies(cash.select_dtypes('object'))
# cash_cats['SK_ID_CURR'] = cash['SK_ID_CURR']
# cash_gb = cash_cats.groupby('SK_ID_CURR').agg('sum').reset_index()
cash_gb.head(2)
### credit card balance data 
ccb = pd.read_csv("../Data/credit_card_balance.csv")
ccb_gb = ccb.groupby("SK_ID_CURR").agg(['min', 'max', 'mean', 'sum', 'var']).add_suffix("_ccb")
ccb_gb.columns = ['_inst_'.join(col) for col in ccb_gb.columns]
ccb_gb = ccb_gb.reset_index()
### Additional hand crafted features
full_train_orig['LOAN_INCOME_RATIO'] = full_train_orig['AMT_CREDIT'] / full_train_orig['AMT_INCOME_TOTAL']
full_train_orig['ANNUITY_INCOME_RATIO'] = full_train_orig['AMT_ANNUITY'] / full_train_orig['AMT_INCOME_TOTAL']
full_train_orig['ANNUITY LENGTH'] = full_train_orig['AMT_CREDIT'] / full_train_orig['AMT_ANNUITY']
full_train_orig['WORKING_LIFE_RATIO'] = full_train_orig['DAYS_EMPLOYED'] / full_train_orig['DAYS_BIRTH']
full_train_orig['INCOME_PER_FAM'] = full_train_orig['AMT_INCOME_TOTAL'] / full_train_orig['CNT_FAM_MEMBERS']
full_train_orig['CHILDREN_RATIO'] = full_train_orig['CNT_CHILDREN'] / full_train_orig['CNT_FAM_MEMBERS']

full_train_orig['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
full_train_orig['DAYS_EMPLOYED_PERC'] = full_train_orig['DAYS_EMPLOYED'] / full_train_orig['DAYS_BIRTH']
full_train_orig['CNT_CHILDREN_outlier'] = (full_train_orig['CNT_CHILDREN'] > 6).astype(int)
for i in full_train_orig['CNT_CHILDREN']:
    if i > 6:
        full_train_orig['CNT_CHILDREN'].replace({i: np.nan}, inplace = True)
        
# full_train_orig['OWN_CAR_AGE_outlier'] = (full_train_orig['OWN_CAR_AGE'] > 60).astype(int)
# for i in full_train_orig['OWN_CAR_AGE']:
#     if i > 60:
#         full_train_orig['OWN_CAR_AGE'].replace({i: np.nan}, inplace = True)

# full_train_orig['CNT_FAM_MEMBERS_outlier'] = (full_train_orig['CNT_FAM_MEMBERS'] > 5).astype(int)
# for i in full_train_orig['CNT_FAM_MEMBERS']:
#     if i > 5:
#         full_train_orig['CNT_FAM_MEMBERS'].replace({i: np.nan}, inplace = True)

full_train_orig['REGION_RATING_CLIENT_W_CITY'].map(lambda s: 1 if s == -1 else 0).sum()
for i in full_train_orig['REGION_RATING_CLIENT_W_CITY']:
    if i == -1:
        full_train_orig['REGION_RATING_CLIENT_W_CITY'].replace({i: 1}, inplace = True)

# full_train_orig['OBS_30_CNT_SOCIAL_CIRCLE_outlier'] = (full_train_orig['OBS_30_CNT_SOCIAL_CIRCLE'] > 17).astype(int)
# for i in full_train_orig['OBS_30_CNT_SOCIAL_CIRCLE']:
#     if i > 17:
#         full_train_orig['OBS_30_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)
#full_train_orig['NEW_CREDIT_TO_GOODS_RATIO'] = full_train_orig['AMT_CREDIT'] / full_train_orig['AMT_GOODS_PRICE']
#full_train_orig = full_train_orig.drop(["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"], axis=1)
# full_train_orig['NEW_INC_PER_CHLD'] = full_train_orig['AMT_INCOME_TOTAL'] / (1 + full_train_orig['CNT_CHILDREN'])
# full_train_orig['NEW_SOURCES_PROD'] = full_train_orig['EXT_SOURCE_1'] * full_train_orig['EXT_SOURCE_2'] * full_train_orig['EXT_SOURCE_3']
# full_train_orig['NEW_EXT_SOURCES_MEAN'] = full_train_orig[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
# full_train_orig['NEW_SCORES_STD'] = full_train_orig[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
# full_train_orig['NEW_CAR_TO_BIRTH_RATIO'] = full_train_orig['OWN_CAR_AGE'] / full_train_orig['DAYS_BIRTH']
# full_train_orig['NEW_CAR_TO_EMPLOY_RATIO'] = full_train_orig['OWN_CAR_AGE'] / full_train_orig['DAYS_EMPLOYED']
# full_train_orig['NEW_PHONE_TO_BIRTH_RATIO'] = full_train_orig['DAYS_LAST_PHONE_CHANGE'] / full_train_orig['DAYS_BIRTH']
# full_train_orig['NEW_PHONE_TO_EMPLOY_RATIO'] = full_train_orig['DAYS_LAST_PHONE_CHANGE'] / full_train_orig['DAYS_EMPLOYED']


test['LOAN_INCOME_RATIO'] = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']
test['ANNUITY_INCOME_RATIO'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']
test['ANNUITY LENGTH'] = test['AMT_CREDIT'] / test['AMT_ANNUITY']
test['WORKING_LIFE_RATIO'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']
test['INCOME_PER_FAM'] = test['AMT_INCOME_TOTAL'] / test['CNT_FAM_MEMBERS']
test['CHILDREN_RATIO'] = test['CNT_CHILDREN'] / test['CNT_FAM_MEMBERS']

test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
test['DAYS_EMPLOYED_PERC'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']
test['CNT_CHILDREN_outlier'] = (test['CNT_CHILDREN'] > 6).astype(int)
for i in test['CNT_CHILDREN']:
    if i > 6:
        test['CNT_CHILDREN'].replace({i: np.nan}, inplace = True)

# test['OWN_CAR_AGE_outlier'] = (test['OWN_CAR_AGE'] > 60).astype(int)
# for i in test['OWN_CAR_AGE']:
#     if i > 60:
#         test['OWN_CAR_AGE'].replace({i: np.nan}, inplace = True)

# test['CNT_FAM_MEMBERS_outlier'] = (test['CNT_FAM_MEMBERS'] > 5).astype(int)
# for i in test['CNT_FAM_MEMBERS']:
#     if i > 5:
#         test['CNT_FAM_MEMBERS'].replace({i: np.nan}, inplace = True)
        
test['REGION_RATING_CLIENT_W_CITY'].map(lambda s: 1 if s == -1 else 0).sum()
for i in test['REGION_RATING_CLIENT_W_CITY']:
    if i == -1:
        test['REGION_RATING_CLIENT_W_CITY'].replace({i: 1}, inplace = True)
        
        
# test['OBS_30_CNT_SOCIAL_CIRCLE_outlier'] = (test['OBS_30_CNT_SOCIAL_CIRCLE'] > 17).astype(int)
# for i in test['OBS_30_CNT_SOCIAL_CIRCLE']:
#     if i > 17:
#         test['OBS_30_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)
#test['NEW_CREDIT_TO_GOODS_RATIO'] = test['AMT_CREDIT'] / test['AMT_GOODS_PRICE']
#test = test.drop(["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"], axis=1)
# test['NEW_INC_PER_CHLD'] = test['AMT_INCOME_TOTAL'] / (1 + test['CNT_CHILDREN'])
# test['NEW_SOURCES_PROD'] = test['EXT_SOURCE_1'] * test['EXT_SOURCE_2'] * test['EXT_SOURCE_3']
# test['NEW_EXT_SOURCES_MEAN'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
# test['NEW_SCORES_STD'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
# test['NEW_CAR_TO_BIRTH_RATIO'] = test['OWN_CAR_AGE'] / test['DAYS_BIRTH']
# test['NEW_CAR_TO_EMPLOY_RATIO'] = test['OWN_CAR_AGE'] / test['DAYS_EMPLOYED']
# test['NEW_PHONE_TO_BIRTH_RATIO'] = test['DAYS_LAST_PHONE_CHANGE'] / test['DAYS_BIRTH']
# test['NEW_PHONE_TO_EMPLOY_RATIO'] = test['DAYS_LAST_PHONE_CHANGE'] / test['DAYS_EMPLOYED']
full_train_orig = pd.merge(full_train_orig, bureau_gb, how="inner", on="SK_ID_CURR")
test = pd.merge(test, bureau_gb, how="left", on="SK_ID_CURR")
full_train_orig.shape
full_train_orig = pd.merge(full_train_orig, prv_gb, how="inner", on="SK_ID_CURR")
test = pd.merge(test, prv_gb, how="left", on="SK_ID_CURR")
full_train_orig = pd.merge(full_train_orig, inst_gb, how="inner", on="SK_ID_CURR")
test = pd.merge(test, inst_gb, how="left", on="SK_ID_CURR")
full_train_orig.shape
full_train_orig = pd.merge(full_train_orig, ccb_gb, how="left", on="SK_ID_CURR")
test = pd.merge(test, ccb_gb, how="left", on="SK_ID_CURR")
full_train_orig.shape
# full_train_orig = pd.merge(full_train_orig, cash_gb, how="inner", on="SK_ID_CURR")
# test = pd.merge(test, cash_gb, how="left", on="SK_ID_CURR")
# '''
# Removing columns with more than 100 null values (filters out 62 columns out of 122 columns)
# And not useful (based on EDA) which removes reduces feature size to 43)
# '''

# null_columns = full_train_orig.columns[full_train_orig.isnull().sum().values > 100000].values.tolist()
# correlated_columns = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT_W_CITY',
#                      'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
# useless_columns = [ "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_7"
#                   ,'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
#        'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
#        'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
#        'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
### Manually removing columns which doesn't make sense (based on EDA)
#full_train = full_train_orig.drop(null_columns+useless_columns, axis = 1)
full_train = full_train_orig
full_train_y = full_train.TARGET.values
full_train = full_train.drop(["TARGET"], axis = 1)
full_train = full_train.set_index("SK_ID_CURR")
num_feats = full_train._get_numeric_data().columns.values.tolist()
cat_feats = list(set(full_train.columns.values) - set(num_feats))
## Categorical Features - Train
train_cat= full_train[cat_feats]
train_cat = pd.get_dummies(train_cat)

## Numerical Features - Train
train_num = full_train[num_feats]
## Categorical Features - Test
test_cat = test[cat_feats]
test_cat = pd.get_dummies(test_cat)

## Numerical Features - Test
test_num = test[num_feats]
full_train_feats = pd.concat([train_num, train_cat], axis=1)
test_feats = pd.concat([test_num, test_cat], axis=1)
full_train_feats = full_train_feats.fillna((full_train_feats.median()))
test_feats = test_feats.fillna(test_feats.median())
# full_train_feats = full_train_feats.apply(lambda x: x.fillna(x.mean()),axis=0)
# test_feats = test_feats.apply(lambda x: x.fillna(x.mean()),axis=0)
train_X, valid_X, train_y, valid_y = train_test_split(full_train_feats, full_train_y, train_size = 0.8, stratify=full_train_y, random_state=42)
# ### RF classifier
# params_rf={
#     'max_depth': [20, 40, 60], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env
#     'n_estimators': [100, 300, 500], #[1000,2000,3000]
# }

# rf_clf = RandomForestClassifier()
# rf = GridSearchCV(rf_clf,
#                   params_rf,
#                   cv=3,
#                   scoring="roc_auc",
#                   n_jobs=1,
#                   verbose=2)
# rf.fit(train_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1), train_y)
# best_est_rf = rf.best_estimator_
# print(best_est)
valid_probs_rf = rf.predict_proba(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))[:,1]
valid_preds_rf = rf.predict(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))
print(accuracy_score(valid_y, valid_preds_rf))
print(roc_auc_score(valid_y, valid_probs_rf))
list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist()))
# params={
#     'max_depth': [3, 5], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env
#     'subsample': [0.6, 0.8], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
#     'colsample_bytree': [0.5, 0.7], #[0.5,0.6,0.7,0.8],
#     'n_estimators': [500, 700], #[1000,2000,3000]
#     'reg_alpha': [0.1, 0.05],  #[0.01, 0.02, 0.03, 0.04]
#     'scale_pos_weight':[3, 5]
# }

# xgb_clf = xgb.XGBClassifier(missing=9999999999)
# rs = GridSearchCV(xgb_clf,
#                   params,
#                   cv=3,
#                   scoring="roc_auc",
#                   n_jobs=1,
#                   verbose=2)
# rs.fit(train_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1), train_y)
# best_est = rs.best_estimator_
# print(best_est)
valid_probs_rs = rs.predict_proba(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))[:,1]
valid_preds_rs= rs.predict(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))[:,1]
print(accuracy_score(valid_y, valid_preds_rs))
print(roc_auc_score(valid_y, valid_probs_rs))
xgb_single = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=9999999999, n_estimators=500,
       nthread=-1, objective='binary:logistic', reg_alpha=0.05,
       reg_lambda=1, scale_pos_weight=3, seed=0, silent=True,
       subsample=0.8)

xgb_single.fit(train_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1), train_y)
valid_probs_xgb_single = xgb_single.predict_proba(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))[:,1]
valid_preds_xgb_single = xgb_single.predict(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))
print(accuracy_score(valid_y, valid_preds_xgb_single))
print(roc_auc_score(valid_y, valid_probs_xgb_single))
### Train AUC
train_probs_xgb_single = xgb_single.predict_proba(train_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))[:,1]
print(roc_auc_score(train_y, train_probs_xgb_single))
xgb_single.fit(full_train_feats.drop(list(set(full_train_feats.columns.tolist()) - set(test_feats.columns.tolist())), axis=1), full_train_y)

params={
    'max_depth': [3, 4, 5], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env
    'subsample': [0.4, 0.6], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5, 0.7], #[0.5,0.6,0.7,0.8],
    'n_estimators': [500, 700], #[1000,2000,3000]
    'reg_alpha': [0.01, 0.05], #[0.01, 0.02, 0.03, 0.04]
    'scale_pos_weight':[3, 5], 
    'num_leaves':[30, 50]
    
}

lgb_clf = lgb.LGBMClassifier()
rs = GridSearchCV(lgb_clf,
                  params,
                  cv=3,
                  scoring="roc_auc",
                  n_jobs=1,
                  verbose=1)
rs.fit(train_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1), train_y)
best_est = rs.best_estimator_
print(best_est)
valid_probs_rs = rs.predict_proba(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))[:,1]
valid_preds_rs= rs.predict(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))
print(accuracy_score(valid_y, valid_preds_rs))
print(roc_auc_score(valid_y, valid_probs_rs))
rs.best_estimator_
best_model = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.5,
        learning_rate=0.1, max_depth=3, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=500,
        n_jobs=-1, num_leaves=30, objective=None, random_state=None,
        reg_alpha=0.05, reg_lambda=0.0, scale_pos_weight=3, silent=True,
        subsample=0.4, subsample_for_bin=200000, subsample_freq=0)

best_model.fit(train_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1), train_y)
valid_probs_best = best_model.predict_proba(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))[:,1]
valid_preds_best = best_model.predict(valid_X.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1))
print(accuracy_score(valid_y, valid_preds_best))
print(roc_auc_score(valid_y, valid_probs_best))
best_model.fit(full_train_feats.drop(list(set(train_X.columns.tolist()) - set(test_feats.columns.tolist())), axis=1), full_train_y)
### Prepare submission file and save to disk
result_df = pd.DataFrame({'SK_ID_CURR':test.SK_ID_CURR.values, "TARGET":xgb_single.predict_proba(test_feats.drop(list(set(test_feats.columns.tolist()) - set(train_X.columns.tolist())), axis=1))[:,1]})
result_df.to_csv("test_submission.csv", index=False)