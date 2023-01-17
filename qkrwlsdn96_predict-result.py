import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold,GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
# from lightgbm import LGBMClassifier as lgb

# import optuna.integration.lightgbm as lgb

# from imblearn.over_sampling import SMOTE
SEED = 1996
train = pd.read_csv('/kaggle/input/finance/train.csv')
test = pd.read_csv('/kaggle/input/finance/test.csv')
sub = pd.read_csv('/kaggle/input/finance/sample.csv')
new_train = pd.read_csv('../input/newdata/new_train_over998.csv')
###### TRAIN #########
mask2 = (train['base_ym'] == 201911)
train=train[mask2]
train.reset_index(inplace = True)
train0 = new_train[new_train['target']==0]
train1 = new_train[new_train['target']==1]
train2 = new_train[new_train['target']==2]
train1 = train1.sample(n = 2000,random_state = 1)
train0 = train0.sample(n = 2111,random_state = 1)
train2 = train2.sample(n = 2500,random_state = 1)
new_train = pd.concat([train0, train1,train2])
new_train.reset_index(inplace = True, drop = True)
train_test = train[train['base_ym'] != 201911]
train_test.reset_index(inplace = True, drop = True)

train_test = train_test[train_test['base_ym'] != 201912]
train_test.reset_index(inplace = True, drop = True)
all_df = pd.concat([train, test])
all_df.reset_index(inplace = True, drop = True)
train.shape, test.shape, all_df.shape
model_var = ['nur_hosp_yn', 'ac_ctr_diff', 
        'fds_cust_yn',
       'hspz_dys_s', 'inamt_nvcd', 
        'dsas_avg_diag_bilg_isamt_s', 'dsas_acd_rst_dcd', 'dsas_ltwt_gcd',
          'optt_blcnt_s', 'base_ym',
       'mtad_cntr_yn', 'heltp_pf_ntyn', 'prm_nvcd', 'surop_blcnt_s',
       'mdct_inu_rclss_dcd', 'dsas_avg_optt_bilg_isamt_s', 'isrd_age_dcd',
       'hspz_blcnt_s', 'dsas_avg_surop_bilg_isamt_s', 'urlb_fc_yn',
       'dsas_avg_hspz_bilg_isamt_s', 'smrtg_5y_passed_yn', 'ac_rst_diff',
       'bilg_isamt_s', 'optt_nbtm_s'] + ['kcd_gcd', 'blrs_cd', 'ar_rclss_cd'] + ['hsp_avg_optt_bilg_isamt_s',\
                                                                                 'hsp_avg_diag_bilg_isamt_s', 'hsp_avg_surop_bilg_isamt_s', 'hsp_avg_hspz_bilg_isamt_s']


all_df['ac_dsas'] = (all_df['ac_ctr_diff'])*(all_df['dsas_ltwt_gcd'])
model_var = model_var + ['ac_dsas']
all_df.loc[all_df['ac_ctr_diff'] == 0, 'ac_ctr_diff'] = np.nan
all_df.loc[all_df['ac_rst_diff'] == 0, 'ac_ctr_diff'] = np.nan
all_df.loc[all_df['ac_dsas'] == 0, 'ac_ctr_diff'] = np.nan
log_var = ['bilg_isamt_s', 'hspz_dys_s'] #, 'optt_nbtm_s']

for col in log_var:
    all_df[col] = np.log1p(all_df[col])
from sklearn import preprocessing
all_df['Change_DSAS'] = all_df['kcd_gcd'].apply(str) + '_' + all_df['dsas_acd_rst_dcd'].apply(str)

le = preprocessing.LabelEncoder()
le.fit(all_df['Change_DSAS'])
all_df['Change_DSAS'] = le.transform(all_df['Change_DSAS'])
model_var = model_var + ['Change_DSAS']
all_df.loc[all_df['prm_nvcd'] == 99, 'prm_nvcd'] = np.nan
all_df.loc[all_df['inamt_nvcd'] == 99, 'inamt_nvcd'] = np.nan

all_df['PRM*inamt'] = np.log1p(all_df['prm_nvcd'] * all_df['inamt_nvcd'])

model_var = model_var + ['PRM*inamt']

dct = all_df.groupby(['dsas_ltwt_gcd', 'ac_ctr_diff'])['dsas_avg_surop_bilg_isamt_s'].mean().to_dict()
all_df['NEW_dsas_avg_surop_bilg_isamt_s'] = all_df.set_index(['dsas_ltwt_gcd', 'ac_ctr_diff']).index.map(dct.get)
all_df['DIFF_dsas_avg_surop_bilg_isamt_s'] = all_df['dsas_avg_surop_bilg_isamt_s']-all_df['NEW_dsas_avg_surop_bilg_isamt_s']
model_var = model_var + ['DIFF_dsas_avg_surop_bilg_isamt_s', 'NEW_dsas_avg_surop_bilg_isamt_s']

########################################################################################################################
dct2 = all_df.groupby(['dsas_ltwt_gcd', 'ac_ctr_diff'])['dsas_avg_surop_bilg_isamt_s'].std().to_dict()
all_df['NEW_STD_dsas_avg_surop_bilg_isamt_s'] = all_df.set_index(['dsas_ltwt_gcd', 'ac_ctr_diff']).index.map(dct2.get)


model_var = model_var + ['NEW_STD_dsas_avg_surop_bilg_isamt_s']
dct = all_df.groupby(['dsas_ltwt_gcd', 'ac_ctr_diff'])['dsas_avg_hspz_bilg_isamt_s', 'dsas_avg_optt_bilg_isamt_s', 'dsas_avg_surop_bilg_isamt_s', 'dsas_avg_diag_bilg_isamt_s'].mean().std(axis = 1)
all_df['dsas_STD'] = all_df.set_index(['dsas_ltwt_gcd', 'ac_ctr_diff']).index.map(dct.get)

dct2 = all_df.groupby(['dsas_ltwt_gcd', 'ac_ctr_diff'])['dsas_avg_hspz_bilg_isamt_s', 'dsas_avg_optt_bilg_isamt_s', 'dsas_avg_surop_bilg_isamt_s', 'dsas_avg_diag_bilg_isamt_s'].mean().sum(axis = 1)
all_df['dsas_SUM'] = all_df.set_index(['dsas_ltwt_gcd', 'ac_ctr_diff']).index.map(dct2.get)

dct3 = all_df.groupby(['dsas_ltwt_gcd', 'ac_ctr_diff'])['dsas_avg_hspz_bilg_isamt_s', 'dsas_avg_optt_bilg_isamt_s', 'dsas_avg_surop_bilg_isamt_s', 'dsas_avg_diag_bilg_isamt_s'].mean().max(axis = 1)
all_df['dsas_MAX'] = all_df.set_index(['dsas_ltwt_gcd', 'ac_ctr_diff']).index.map(dct3.get)

dct4 = all_df.groupby(['dsas_ltwt_gcd', 'ac_ctr_diff'])['dsas_avg_hspz_bilg_isamt_s', 'dsas_avg_optt_bilg_isamt_s', 'dsas_avg_surop_bilg_isamt_s', 'dsas_avg_diag_bilg_isamt_s'].mean().min(axis = 1)
all_df['dsas_MIN'] = all_df.set_index(['dsas_ltwt_gcd', 'ac_ctr_diff']).index.map(dct4.get)

all_df['dsas_MAX-MIN'] = all_df['dsas_MAX'] - all_df['dsas_MIN']

model_var = model_var + ['dsas_MIN', 'dsas_MAX', 'dsas_MAX-MIN', 'dsas_SUM', 'dsas_STD']
all_df.loc[all_df['mdct_inu_rclss_dcd'] == 9, 'mdct_inu_rclss_dcd'] = np.nan
train = all_df[all_df['base_ym'] == 201911]
train.reset_index(inplace = True, drop = True)

test = all_df[all_df['base_ym'] == 201912]
test.reset_index(inplace = True, drop = True)

train = pd.concat([train, new_train])
train.reset_index(inplace = True, drop = True)
train = pd.concat([train, new_train])
train.reset_index(inplace = True, drop = True)
NUM_BOOST_ROUND = 50000

#######################
### FOR Stratified ####
N_SPLITS = 5
#######################
#######################

lgbm_param = {
    "objective": "multiclassova",
    'n_estimators' : NUM_BOOST_ROUND,
    "boosting": "gbdt",
    "num_leaves": 50,
    "learning_rate": 0.008,
    "feature_fraction": 0.95,
    "reg_lambda": 2,
    "metric": "multiclass",
    "num_class" : 3,
    'seed' : SEED,
}


final_test = np.zeros(( test.shape[0], 3 ))
lgbm_oof_train = np.zeros((train.shape[0]))


kfolds = StratifiedKFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
for ind, (trn_ind, val_ind) in tqdm( enumerate(kfolds.split(X= train[model_var], y = train['ac_dsas'] )) ):
    X_train , y_train = train.iloc[trn_ind][model_var], train.iloc[trn_ind]['target']
    X_valid , y_valid = train.iloc[val_ind][model_var], train.iloc[val_ind]['target']

#     w = y_train.value_counts()
#     weights = {i : np.sum(w) / w[i] for i in w.index}
    
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid)
    
    # model 정의&학습
    model = lgb.train(lgbm_param , dtrain,  num_boost_round=NUM_BOOST_ROUND,  verbose_eval=2000,  early_stopping_rounds=500,
                       valid_sets=(dtrain, dvalid),
                       valid_names=('train','valid')) #, sample_weight=y_train.map(weights))
    

    lgb.plot_importance(model, importance_type='gain', max_num_features = 30)


    lgbm_valid_pred = model.predict(X_valid)
    lgbm_valid_pred =  list(map(np.argmax, lgbm_valid_pred))    
    lgbm_oof_train[val_ind] = lgbm_valid_pred
    print('='*80)
    
    ## 결국 lgbm으로 제출띠~
    test_predict = model.predict(test[model_var])
    final_test += test_predict
    

    
final_test /= N_SPLITS  
print(f"<Light-GBM> OVERALL : {f1_score( train['target'], lgbm_oof_train, average = None ),f1_score( train['target'], lgbm_oof_train, average = None ).mean() }")
np.save('final_test_998_50_2', final_test)
result =  list(map(np.argmax, final_test))
sub['target'] = result
display(sub['target'].value_counts()/len(sub))
display(train['target'].value_counts()/len(train))
sub.to_csv('./new_train_998_result_pleasE_lef50_v2.csv', index = False)
from IPython.display import FileLink
FileLink(r'result.csv')
