import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')

gc.enable()
print('Importing data...')
lgbm_submission = pd.read_csv('../input/sample_submission.csv')
buro_bal = pd.read_csv('../input/bureau_balance.csv')
print('Buro bal shape : ', buro_bal.shape)

print('transform to dummies')
buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)

print('Counting buros')
buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

print('averaging buro bal')
avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
del buro_bal
gc.collect()

print('Read Bureau')
buro = pd.read_csv('../input/bureau.csv')

print('Go to dummies')
buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
# buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]

del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
gc.collect()

print('Merge with buro avg')
buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))

print('Counting buro per SK_ID_CURR')
nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

print('Averaging bureau')
avg_buro = buro_full.groupby('SK_ID_CURR').mean()
#print(avg_buro.head())

del buro, buro_full
gc.collect()

print('Read prev')
prev = pd.read_csv('../input/previous_application.csv')

prev_cat_features = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]

print('Go to dummies')
prev_dum = pd.DataFrame()
for f_ in prev_cat_features:
    prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

prev = pd.concat([prev, prev_dum], axis=1)

del prev_dum
gc.collect()

print('Counting number of Prevs')
nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

print('Averaging prev')
avg_prev = prev.groupby('SK_ID_CURR').mean()
#print(avg_prev.head())
del prev
gc.collect()

print('Reading POS_CASH')
pos = pd.read_csv('../input/POS_CASH_balance.csv')

print('Go to dummies')
pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

print('Compute nb of prevs per curr')
nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Go to averages')
avg_pos = pos.groupby('SK_ID_CURR').mean()

del pos, nb_prevs
gc.collect()

print('Reading CC balance')
cc_bal = pd.read_csv('../input/credit_card_balance.csv')

print('Go to dummies')
cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Compute average')
avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

del cc_bal, nb_prevs
gc.collect()

print('Reading Installments')
inst = pd.read_csv('../input/installments_payments.csv')
nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

avg_inst = inst.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]

print('Read data and test')
data = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
print('Shapes : ', data.shape, test.shape)

y = data['TARGET']
del data['TARGET']

categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]
categorical_feats
for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])
    
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

del avg_buro, avg_prev
gc.collect()
#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)
#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)
#Select Hyper-Parameters
params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}
#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds= 30,
                 verbose_eval= 10
                 )
y_hat = lgbm.predict(data)
score = roc_auc_score(y, y_hat)
print("Overall AUC: {:.3f}" .format(score))
import shap
%time shap_values = shap.TreeExplainer(lgbm).shap_values(valid_x)
shap.summary_plot(shap_values, valid_x)
shap.dependence_plot("EXT_SOURCE_2", shap_values, valid_x)
shap.dependence_plot("EXT_SOURCE_3", shap_values, valid_x)
shap.dependence_plot("EXT_SOURCE_1", shap_values, valid_x)
shap.dependence_plot("AMT_GOODS_PRICE_x", shap_values, valid_x)
shap.dependence_plot("CODE_GENDER", shap_values, valid_x)
data['CODE_GENDER'].value_counts()
# Who are those four ?
data[data['CODE_GENDER']==2]
# Did those got a loan ?
y[data['CODE_GENDER']==2].describe()
# Do we have those in test as well ?
test['CODE_GENDER'].value_counts()
shap.dependence_plot("AMT_ANNUITY_x", shap_values, valid_x)
shap.dependence_plot("inst_AMT_PAYMENT", shap_values, valid_x)
# How many outliers are they and do they get a loan ?
y[data['inst_AMT_PAYMENT'] > 500000].describe()
# Do those outliers also occur in the test set ?
test[test['inst_AMT_PAYMENT'] > 500000].filter(regex='EXT_SOURCE_.', axis=1).describe()
shap.dependence_plot("AMT_CREDIT_x", shap_values, valid_x)
shap.dependence_plot("DAYS_BIRTH", shap_values, valid_x)
shap.dependence_plot("CNT_INSTALMENT_FUTURE", shap_values, valid_x)
shap.dependence_plot("DAYS_EMPLOYED", shap_values, valid_x)
y[data['DAYS_EMPLOYED'] > 0].describe()
data[data['DAYS_EMPLOYED'] <= 0].DAYS_EMPLOYED.hist()
y[data['DAYS_EMPLOYED'] <= 0].describe()
data[data['DAYS_EMPLOYED'] > 0].filter(regex='EXT_SOURCE_.', axis=1).describe()
test[test['DAYS_EMPLOYED'] <= 0].filter(regex='EXT_SOURCE_.', axis=1).describe()
# how many unemployed in the test set ?
test[test['DAYS_EMPLOYED'] > 0].filter(regex='EXT_SOURCE_.', axis=1).describe()
shap.dependence_plot("CNT_PAYMENT", shap_values, valid_x)
shap.dependence_plot("NAME_EDUCATION_TYPE", shap_values, valid_x)
shap.dependence_plot("SK_ID_PREV_y", shap_values, valid_x)
shap.dependence_plot("NAME_CONTRACT_STATUS_Refused", shap_values, valid_x)
shap.dependence_plot("OWN_CAR_AGE", shap_values, valid_x)
shap.dependence_plot("cc_bal_CNT_DRAWINGS_ATM_CURRENT", shap_values, valid_x)
y[data['cc_bal_CNT_DRAWINGS_ATM_CURRENT'] > 10].describe()
test[test['cc_bal_CNT_DRAWINGS_ATM_CURRENT'] > 10].shape
shap.dependence_plot("AMT_CREDIT_SUM_DEBT", shap_values, valid_x)
shap.dependence_plot("SK_DPD_DEF", shap_values, valid_x)
y[data['SK_DPD_DEF'] > 0].describe()
shap.dependence_plot("DAYS_CREDIT", shap_values, valid_x)
# shap.initjs()
# Makes the browser to slow
# shap.force_plot(shap_values[:1000,:], valid_x.iloc[:1000,:])
sub_pred = lgbm.predict(test)
sub_pred = np.clip(sub_pred, 0, 1)
sub_pred.min(), sub_pred.max()
lgbm_submission.TARGET = sub_pred
lgbm_submission.to_csv('subm_lgbm_auc{:.8f}.csv'.format(score), index=False, float_format='%.8f')
