import pandas as pd

import numpy as np

import random as rnd

import os

# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import lightgbm



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings('ignore')

os.listdir('../input')
#train = pd.read_csv('../input/train-na3/new_na3.csv')

train = pd.read_csv('../input/upsample-train-na1/up_train_na1.csv')

valid =  pd.read_csv('../input/valid-na3/valid_na3.csv')

test = pd.read_csv('../input/test-na2/test_ver_1.csv')

smallE = np.e**-12
def regularization_mean_std_d(data_arr):

    mean = np.mean(data_arr)

    std = np.std(data_arr) 

    regularaz_arr = (data_arr - mean)/std

    return regularaz_arr
def regularization_log_d(data_arr):

    return np.log(data_arr)
def regularization_l2(data_arr):

    return (data_arr/ np.linalg.norm(data_arr))
train_v1 = train
list(train_v1.columns)
train =train.drop(columns=['bldng_us_clssfctn','wnd','emd_nm'])

train =train.drop(columns=['id','dt_of_fr'])

test =test.drop(columns=['id','dt_of_fr','bldng_us_clssfctn','wnd','emd_nm','fr_yn','year'])

valid = valid.drop(columns=['id','dt_of_fr'])

valid = valid.drop(columns=['bldng_us_clssfctn','wnd','emd_nm'])
train_v1 =train_v1.drop(columns=['bldng_us_clssfctn','year','wnd','emd_nm','Class'])

train_v1 =train_v1.drop(columns=['id','dt_of_fr'])
binary_y = {'N': 0, 'Y': 1}

train['fr_yn'] = train['fr_yn'].map(binary_y)

train_v1['fr_yn'] = train_v1['fr_yn'].map(binary_y)



valid['fr_yn'] = valid['fr_yn'].map(binary_y)
for col in train.columns:

    null_count = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (train[col].isnull().sum() / train[col].shape[0]))

    print(null_count)
bldng_us_arr = train['bldng_us'].unique()
f,ax = plt.subplots(1,1,figsize=(18,8))

sns.countplot('bldng_us',hue='fr_yn',data=train)

plt.title("bldng_us")
bldng_us_one_hot = pd.get_dummies(train[train['fr_yn']==1]['bldng_us'])
bldng_us_cols = train['bldng_us'].unique()
bldng_us_cols_dic= {}
for i,one  in enumerate(bldng_us_cols):

    bldng_us_cols_dic[one] = i 
train_v1['bldng_us'] =train_v1['bldng_us'].map(bldng_us_cols_dic)

valid['bldng_us'] = valid['bldng_us'].map(bldng_us_cols_dic)

test['bldng_us'] = test['bldng_us'].map(bldng_us_cols_dic)
train['bldng_archtctr'].value_counts()
f,ax = plt.subplots(1,1,figsize=(18,8))

sns.countplot('bldng_archtctr',hue='fr_yn',data=train)

plt.title("bldng_archtctr")
train['wnd_drctn'].value_counts()
f,ax = plt.subplots(1,1,figsize=(18,8))

sns.countplot('wnd_drctn',hue='fr_yn',data=train)

plt.title("wnd_drctn")
train[train['fr_yn']==1]['wnd_drctn']

train_v1['wnd_drctn'].loc[train_v1['wnd_drctn']!='none'] = pd.to_numeric(train_v1['wnd_drctn'].loc[train_v1['wnd_drctn']!='none'])/180*np.pi

valid['wnd_drctn'].loc[valid['wnd_drctn']!='none'] = pd.to_numeric(valid['wnd_drctn'].loc[valid['wnd_drctn']!='none'])/180*np.pi

train_v1['wnd_drctn'] = train_v1['wnd_drctn'].replace({'none':-1})

valid['wnd_drctn'] = valid['wnd_drctn'].replace({'none':-1})

test['wnd_drctn'].loc[test['wnd_drctn']!='none'] = pd.to_numeric(test['wnd_drctn'].loc[test['wnd_drctn']!='none'])/90

test['wnd_drctn'] = test['wnd_drctn'].replace({'none':-1})
train_v1['wnd_drctn'].head()
sns.kdeplot(train['wnd_spd'])
f,ax = plt.subplots(1,1,figsize=(18,8))

sns.countplot('bldng_cnt_in_50m',hue='fr_yn',data=train)

plt.title("bldng_cnt_in_50m")
pd.crosstab(train['bldng_cnt_in_50m'],train['fr_yn']==1).style.background_gradient(cmap="summer_r")
#train['bldng_cnt_in_50m'] = regularization_mean_std_d(train['bldng_cnt_in_50m'])

train_v1['bldng_cnt_in_50m'] = regularization_log_d(train_v1['bldng_cnt_in_50m']+smallE)

valid['bldng_cnt_in_50m'] = regularization_log_d(valid['bldng_cnt_in_50m']+smallE)

test['bldng_cnt_in_50m'] = regularization_log_d(test['bldng_cnt_in_50m']+smallE)
sns.kdeplot(regularization_log_d(train['bldng_cnt_in_50m']+smallE))
sns.kdeplot(train[train['fr_yn']==1]['no_tbc_zn_dstnc'])
sns.kdeplot(regularization_log_d(train['no_tbc_zn_dstnc']))
train_v1['no_tbc_zn_dstnc']= regularization_log_d(train_v1['no_tbc_zn_dstnc']+smallE)

valid['no_tbc_zn_dstnc']= regularization_log_d(valid['no_tbc_zn_dstnc']+smallE)

test['no_tbc_zn_dstnc']= regularization_log_d(test['no_tbc_zn_dstnc']+smallE)
sns.kdeplot(regularization_mean_std_d(train['sft_emrgnc_bll_dstnc']))
sft_emrgnc_bll_dstnc_arr = regularization_mean_std_d(np.asarray(train[train['fr_yn']==1]['sft_emrgnc_bll_dstnc']))

train_v1['sft_emrgnc_bll_dstnc'] = regularization_mean_std_d(train_v1['sft_emrgnc_bll_dstnc']+smallE)

valid['sft_emrgnc_bll_dstnc'] = regularization_mean_std_d(valid['sft_emrgnc_bll_dstnc']+smallE)

test['sft_emrgnc_bll_dstnc'] = regularization_mean_std_d(test['sft_emrgnc_bll_dstnc']+smallE)

sns.kdeplot(sft_emrgnc_bll_dstnc_arr)
sns.kdeplot(train_v1['hmdt'])
train_v1['hmdt'] = regularization_mean_std_d(train_v1['hmdt'])

test['hmdt'] = regularization_mean_std_d(test['hmdt'])
sns.kdeplot(regularization_log_d(train['fr_wthr_fclt_dstnc']))
fr_wthr_fclt_dstnc_arr = (np.asarray(train[train['fr_yn']==1]['fr_wthr_fclt_dstnc']))

fr_wthr_fclt_dstnc_arr_d_s = regularization_log_d(fr_wthr_fclt_dstnc_arr)

sns.kdeplot(fr_wthr_fclt_dstnc_arr_d_s)

train_v1['fr_wthr_fclt_dstnc'] = regularization_log_d(train_v1['fr_wthr_fclt_dstnc']+smallE)

valid['fr_wthr_fclt_dstnc'] = regularization_log_d(valid['fr_wthr_fclt_dstnc']+smallE)

test['fr_wthr_fclt_dstnc'] = regularization_log_d(test['fr_wthr_fclt_dstnc']+smallE)

sns.kdeplot((train['cctv_in_100m']))
cctv_in_100m_reg_log =regularization_log_d(np.asarray(train[train['fr_yn']==1]['cctv_in_100m']))
train_v1['cctv_in_100m'] = regularization_log_d(train_v1['cctv_in_100m']+smallE)

valid['cctv_in_100m'] = regularization_log_d(valid['cctv_in_100m']+smallE)

test['cctv_in_100m'] = regularization_log_d(test['cctv_in_100m']+smallE)
sns.kdeplot(np.asarray(train[train['fr_yn']==1]['ahsm_dstnc']))
ahsm_dstnc_mean_std_d = regularization_mean_std_d(train[train['fr_yn']==1]['ahsm_dstnc'])
sns.kdeplot(ahsm_dstnc_mean_std_d)

train_v1['ahsm_dstnc'] = regularization_mean_std_d(train_v1['ahsm_dstnc']+smallE)

valid['ahsm_dstnc'] = regularization_mean_std_d(valid['ahsm_dstnc']+smallE)

test['ahsm_dstnc'] = regularization_mean_std_d(test['ahsm_dstnc']+smallE)

#binary_y = {'N': 0, 'Y': 1}

train['mlt_us_yn'] = train['mlt_us_yn'].map(binary_y)

train_v1['mlt_us_yn'] = train_v1['mlt_us_yn'].map(binary_y)

valid['mlt_us_yn'] = valid['mlt_us_yn'].map(binary_y)

test['mlt_us_yn'] = test['mlt_us_yn'].map(binary_y)
train_v1.drop(columns=['mlt_us_yn'])

train.drop(columns=['mlt_us_yn'])

valid.drop(columns=['mlt_us_yn'])

test.drop(columns=['mlt_us_yn'])
f,ax = plt.subplots(1,1,figsize=(18,8))

sns.countplot('mlt_us_yn',hue='fr_yn',data=train[train['fr_yn']==1])

plt.title("mlt_us_yn")
#train['fr_mn_cnt'].head()

sns.kdeplot(train['fr_mn_cnt'])
fr_mn_cn_std = regularization_mean_std_d(train[train['fr_yn']==1]['fr_mn_cnt'])
train_v1['fr_mn_cnt'] = regularization_mean_std_d(train_v1['fr_mn_cnt'])
sns.kdeplot(fr_mn_cn_std)
sns.kdeplot(regularization_log_d(train['lnd_ar']+smallE))
lnd_ar_cos_d = np.cos(train['lnd_ar'])
sns.kdeplot(lnd_ar_cos_d)

train_v1['lnd_ar'] = regularization_log_d(train_v1['lnd_ar']+smallE)

valid['lnd_ar'] = regularization_log_d(valid['lnd_ar']+smallE)

test['lnd_ar'] = regularization_log_d(test['lnd_ar']+smallE)
sns.kdeplot(train_v1['ttl_ar'])
train_v1['ttl_ar'] =  regularization_log_d(train_v1['ttl_ar'] +smallE)

valid['ttl_ar'] =  regularization_log_d(valid['ttl_ar'] +smallE)

test['ttl_ar'] = regularization_log_d(test['ttl_ar'] +smallE)
sns.kdeplot(train_v1['ttl_ar'])
sns.kdeplot(train[train['fr_yn']==1]['bldng_ar'])
bldng_ar_log_d = regularization_log_d(train['bldng_ar'])
sns.kdeplot(bldng_ar_log_d)

train_v1['bldng_ar'] = regularization_log_d(train_v1['bldng_ar']+ smallE)

valid['bldng_ar'] = regularization_log_d(valid['bldng_ar']+ smallE)

test['bldng_ar'] = regularization_log_d(test['bldng_ar']+ smallE)
sns.kdeplot(regularization_log_d(train_v1['floor']))

train_v1['floor'] = regularization_log_d(train_v1['floor'])

#train_v1 =train_v1.drop(columns=['ttl_ar','bldng_ar'])

train_v1 = train_v1.drop(columns =['floor'])

valid =  valid.drop(columns =['floor'])

test=  test.drop(columns =['floor'])
sns.kdeplot((train['tbc_rtl_str_dstnc']))
train_v1['tbc_rtl_str_dstnc'] = regularization_log_d(train_v1['tbc_rtl_str_dstnc']+smallE)

valid['tbc_rtl_str_dstnc'] = regularization_log_d(valid['tbc_rtl_str_dstnc']+smallE)

test['tbc_rtl_str_dstnc'] = regularization_log_d(test['tbc_rtl_str_dstnc']+smallE)
#train_v1['dt_of_athrztn']= pd.to_numeric(train_v1['dt_of_athrztn']) - 2000

#valid['dt_of_athrztn']= pd.to_numeric(valid['dt_of_athrztn']) - 2000

train_v1['dt_of_athrztn'] = regularization_log_d(train['dt_of_athrztn'])

valid['dt_of_athrztn'] = regularization_log_d(valid['dt_of_athrztn'])

test['dt_of_athrztn'] = regularization_log_d(test['dt_of_athrztn'])
train.iloc[43]
bldng_archtctr_one_hot_df = pd.get_dummies(train_v1['bldng_archtctr'])

valid_bldng_archtctr_one_hot_df  = pd.get_dummies(valid['bldng_archtctr'])

test_bldng_archtctr_one_hot_df = pd.get_dummies(test['bldng_archtctr'])
valid_bldng_archtctr_one_hot_df['프리케스트콘크리트구조']=0
len(bldng_archtctr_one_hot_df.columns)
#train_v1 = pd.concat([train_v1,bldng_archtctr_one_hot_df],axis=1)

#valid = pd.concat([valid,valid_bldng_archtctr_one_hot_df],axis=1)
#train_v1 = pd.concat([train_v1,bldng_archtctr_one_hot_df],axis=1)

#valid = pd.concat([valid,valid_bldng_archtctr_one_hot_df],axis=1)

bldng_archtctr_arr=train_v1['bldng_archtctr'].unique()

bldng_archtctr_dic = {}

for i,one in enumerate(bldng_archtctr_arr):

    bldng_archtctr_dic[one]=i+1
train_v1['bldng_archtctr'] = train_v1['bldng_archtctr'].map(bldng_archtctr_dic)

valid['bldng_archtctr'] = valid['bldng_archtctr'].map(bldng_archtctr_dic)

test['bldng_archtctr'] = test['bldng_archtctr'].map(bldng_archtctr_dic)

#train_v1 =train_v1.drop(columns=['bldng_archtctr'])

#valid = valid.drop(columns=['bldng_archtctr'])
sns.countplot(train_v1['jmk'])
jmk_unique = train_v1['jmk'].unique()
jmk_dic = {}

jmk_bias = -int(len(jmk_unique)/2-1)
for i,one in enumerate(jmk_unique):

    jmk_dic[one] = jmk_bias+i
train_v1['jmk'] = train_v1['jmk'].map(jmk_dic)

valid['jmk'] = valid['jmk'].map(jmk_dic)

test['jmk'] = test['jmk'].map(jmk_dic)
rgnl_ar_nm_unique_arr = train_v1['rgnl_ar_nm'].unique()

rgnl_ar_nm_dic = {}
for i,one in enumerate(rgnl_ar_nm_unique_arr):

    rgnl_ar_nm_dic[one] = -i-1
train_v1['rgnl_ar_nm'] = train_v1['rgnl_ar_nm'].map(rgnl_ar_nm_dic)

valid['rgnl_ar_nm'] = valid['rgnl_ar_nm'].map(rgnl_ar_nm_dic)

test['rgnl_ar_nm'] = test['rgnl_ar_nm'].map(rgnl_ar_nm_dic)
rgnl_ar_nm2_unique_arr = train_v1['rgnl_ar_nm2'].unique()

rgnl_ar_nm2_dic = {}
for i,one in enumerate(rgnl_ar_nm2_unique_arr):

    rgnl_ar_nm2_dic[one] = i+1
train_v1['rgnl_ar_nm2'] = train_v1['rgnl_ar_nm2'].map(rgnl_ar_nm2_dic)

valid['rgnl_ar_nm2'] = valid['rgnl_ar_nm2'].map(rgnl_ar_nm2_dic)

test['rgnl_ar_nm2'] = test['rgnl_ar_nm2'].map(rgnl_ar_nm2_dic)
lnd_us_sttn_nm_unique = train_v1['lnd_us_sttn_nm'].unique()

lnd_us_sttn_nm_dic = {}
for i,one in enumerate(lnd_us_sttn_nm_unique):

    lnd_us_sttn_nm_dic[one]= i+1
train_v1['lnd_us_sttn_nm'] = train_v1['lnd_us_sttn_nm'].map(lnd_us_sttn_nm_dic)

valid['lnd_us_sttn_nm'] = valid['lnd_us_sttn_nm'].map(lnd_us_sttn_nm_dic)

test['lnd_us_sttn_nm'] = valid['lnd_us_sttn_nm'].map(lnd_us_sttn_nm_dic)
sns.kdeplot(regularization_log_d(train_v1['hm_cnt']))
sns.kdeplot(regularization_log_d(train_v1['hm_cnt']))
train_v1['bldng_ar_prc'] = (regularization_log_d(train_v1['bldng_ar_prc'] +smallE))

valid['bldng_ar_prc'] = (regularization_log_d(valid['bldng_ar_prc']+smallE))

test['bldng_ar_prc'] = (regularization_log_d(test['bldng_ar_prc']+smallE))
rd_sd_nm_arr = train_v1['rd_sd_nm'].unique()

rd_sd_nm_dic = {}
for i,one in enumerate(rd_sd_nm_arr):

    rd_sd_nm_dic[one]=i+1
train_v1['rd_sd_nm'] = train_v1['rd_sd_nm'].map(rd_sd_nm_dic)

valid['rd_sd_nm'] = valid['rd_sd_nm'].map(rd_sd_nm_dic)

test['rd_sd_nm'] = test['rd_sd_nm'].map(rd_sd_nm_dic)
sns.kdeplot((train_v1['fr_sttn_dstnc']))
sns.kdeplot(regularization_log_d(train_v1['fr_sttn_dstnc']))
train_v1['fr_sttn_dstnc'] =regularization_log_d(train_v1['fr_sttn_dstnc']+smallE)

valid['fr_sttn_dstnc'] =regularization_log_d(valid['fr_sttn_dstnc']+smallE)

test['fr_sttn_dstnc'] =regularization_log_d(test['fr_sttn_dstnc']+smallE)
ele = 'ele_engry_us_'

gas = 'gas_engry_us_'

gas_engry_us_cols = []

ele_engry_us_cols = []

for one in train_v1.columns:

    if(ele in one):

        ele_engry_us_cols.append(one)

    elif(gas in one):

        gas_engry_us_cols.append(one)
for one in ele_engry_us_cols:

    train_v1[one]=regularization_log_d( train_v1[one]+smallE)

    valid[one]=regularization_log_d( valid[one]+smallE)

    test[one]=regularization_log_d( test[one]+smallE)
for one in gas_engry_us_cols:

    train_v1[one]=regularization_log_d( train_v1[one]+smallE)

    valid[one]=regularization_log_d( valid[one]+smallE)

    test[one]=regularization_log_d( test[one]+smallE)
#gas_engry_us_df = train_v1[gas_engry_us_cols]

#ele_engry_us_df = train_v1[ele_engry_us_cols]

#drop_ele_gas = ele_engry_us_cols

#drop_ele_gas.extend(gas_engry_us_df)

#train_v1 = train_v1.drop(columns=drop_ele_gas)

#valid_gas_engry_us_df = valid[gas_engry_us_cols]

#valid_ele_engry_us_df = valid[ele_engry_us_cols]

#valid = valid.drop(columns=drop_ele_gas)
from keras import *

from keras.models import *
train_v1[train_v1['fr_yn']==1].count()[0]
train_v1[train_v1['fr_yn']==0].count()[1]
(train_v1[train_v1['fr_yn']==0].count()[1] +train_v1[train_v1['fr_yn']==1].count()[0]) == train_v1.count()[0]
fire_no = list(train_v1.index[train_v1['fr_yn']==0][:train_v1[train_v1['fr_yn']==1].count()[0]])

fire_yes = list(train_v1.index[train_v1['fr_yn']==1][:train_v1[train_v1['fr_yn']==1].count()[0]])
#train_v1 = pd.concat([train_v1.iloc[fire_no],train_v1.iloc[fire_yes]])

#train_v1 =shuffle(train_v1)
import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import metrics   #Additional scklearn functions

from sklearn.model_selection import GridSearchCV   #Perforing grid search

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

from sklearn.metrics import f1_score

from keras import optimizers

from sklearn.utils import shuffle
train_y_v1 = train_v1['fr_yn']

train_v1= train_v1.drop(columns=['fr_yn'])

valid_y =valid['fr_yn']

valid_v1= valid.drop(columns=['fr_yn'])

X_tr, X_vld, y_tr, y_vld  = train_test_split(train_v1, train_y_v1, test_size=0.2, random_state=2019)
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

#ada.fit(train_v1, train_y_v1)
print(len(train_v1.columns))

print(len(valid_v1.columns))

print(len(test.columns))
xgb1 =  XGBClassifier(

    learning_rate =0.04,

    n_estimators=150,

    max_depth=20,

    min_child_weight=9,

    gamma=0.0001,

    reg_alpha=5e-06,

    subsample=0.77,

    colsample_bytree=0.8,

    objective= 'binary:logistic',

    nthread=-1,

    scale_pos_weight=1,

    seed=2019

)
#xgb1.fit(train_v1,train_y_v1)

xgb1.fit(X_tr,y_tr)
#xgb1.score(valid_v1,valid_y)

xgb1.score(X_vld,y_vld)

#0.8325344952795933
#res_valid =xgb1.predict(valid_v1)

x_res_valid =xgb1.predict(X_vld)
#f1_score(valid_y,res_valid)

f1_score(y_vld,x_res_valid)

#0.42896764252696457
xgb2 =  XGBClassifier(

    learning_rate =0.05,

    n_estimators=150,

    max_depth=14,

    min_child_weight=9,

    gamma=0.0001,

    reg_alpha=5e-06,

    subsample=0.77,

    colsample_bytree=0.8,

    objective= 'binary:logistic',

    nthread=-1,

    scale_pos_weight=1,

    seed=2018

)
xgb2.fit(train_v1,train_y_v1)
xgb2.score(valid_v1,valid_y)

#xgb1.score(X_vld,y_vld)
res_valid_2 =xgb2.predict(valid_v1)

#x_res_valid =xgb1.predict(X_vld)
f1_score(valid_y,res_valid_2)

#f1_score(y_vld,x_res_valid)

#0.42896764252696457
xgb3 = XGBClassifier(

    learning_rate =0.04,

    n_estimators=150,

    max_depth=12,

    min_child_weight=8,

    gamma=0.005,

    reg_alpha=1e-05,

    subsample=0.99,

    colsample_bytree=0.9,

    objective= 'binary:logistic',

    nthread=-1,

    scale_pos_weight=1,

    seed=2016

)
xgb3.fit(train_v1,train_y_v1)
xgb3.score(valid_v1,valid_y)
res_valid_3 =xgb3.predict(valid_v1)
f1_score(valid_y,res_valid_3)
xgb4 = XGBClassifier(

    learning_rate =0.05,

    n_estimators=200,

    max_depth=10,

    min_child_weight=8,

    gamma=0.001,

    reg_alpha=1e-03,

    subsample=0.99,

    colsample_bytree=0.7,

    objective= 'binary:logistic',

    nthread=-1,

    scale_pos_weight=1,

    seed=2017

)
xgb4.fit(train_v1,train_y_v1)
res_valid_4 = xgb4.predict(valid_v1)
f1_score(valid_y,res_valid_4)
def predict_test(test):

    test_res = (xgb1.predict(test)+xgb2.predict(test)+xgb3.predict(test)+xgb4.predict(test))/4

    test_res = test_res.round()

    return test_res
ensem_res = predict_test(valid_v1)
f1_score(valid_y,ensem_res.round())
xgb5 = XGBClassifier(

    learning_rate =0.05,

    n_estimators=200,

    max_depth=10,

    min_child_weight=8,

    gamma=0.005,

    reg_alpha=1e-03,

    subsample=0.99,

    colsample_bytree=0.7,

    objective= 'binary:logistic',

    nthread=-1,

    scale_pos_weight=1,

    seed=2015

)
xgb5.fit(train_v1,train_y_v1)



res_valid_5 = xgb5.predict(valid_v1)

f1_score(valid_y,res_valid_5)
test.tail()
PJT002_submission = pd.read_csv("../input/submission-csv/PJT002_submission.csv")

PJT002_submission['fr_yn'] = predict_test(test)

#predict_test

#xgb1.predict
PJT002_submission.head()
binary_y = {0.0 :'N', 1.0:'Y'}

PJT002_submission['fr_yn'] = PJT002_submission['fr_yn'].map(binary_y)
PJT002_submission.to_csv("./PJT002_submission.csv", header=True, index=False)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3,30,100],'max_features' : [2,4,6,8]},

    {'bootstrap':[False], 'n_estimators':[3,100],'max_features':[2,3,4]},

]



grid_search = GridSearchCV(xgb5,param_grid,cv=5,

                          scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(train_v1,train_y_v1)

feature_importances = grid_search.best_estimator_.feature_importances_



res_valid_6 = grid_search.predict(valid_v1)

f1_score(valid_y,res_valid_6)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3,8,100],'max_features' : [2,4,6,8]},

    {'bootstrap':[False], 'n_estimators':[3,100],'max_features':[2,3,4]},

]



grid_search = GridSearchCV(xgb2,param_grid,cv=5,

                          scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(train_v1,train_y_v1)

feature_importances = grid_search.best_estimator_.feature_importances_



res_valid_7 = grid_search.predict(valid_v1)

f1_score(valid_y,res_valid_7)
param_grid = [

    {'n_estimators': [3,8,100],'max_features' : [2,4,6,8]},

    {'bootstrap':[False], 'n_estimators':[3,100],'max_features':[2,3,4]},

]



grid_search = GridSearchCV(xgb4,param_grid,cv=5,

                          scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(train_v1,train_y_v1)

feature_importances = grid_search.best_estimator_.feature_importances_



res_valid_8 = grid_search.predict(valid_v1)

f1_score(valid_y,res_valid_8)
grid_search.best_estimator_
test_res = (res_valid_2+res_valid_7+res_valid_6+res_valid_3)/4

test_res = test_res.round()

f1_score(valid_y,test_res)
