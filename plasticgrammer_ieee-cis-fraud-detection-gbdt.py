import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc,os,sys

import re

import random



from sklearn import metrics, preprocessing

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut

from sklearn.decomposition import PCA, KernelPCA, NMF

from sklearn.cluster import KMeans

from tqdm import tqdm



import warnings

warnings.filterwarnings('ignore')



sns.set_style('darkgrid')



pd.options.display.float_format = '{:,.3f}'.format



print(os.listdir("../input"))
def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



seed_everything(42)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
%%time

train_id = pd.read_csv('../input/train_identity.csv')

train_trn = pd.read_csv('../input/train_transaction.csv')

test_id = pd.read_csv('../input/test_identity.csv')

test_trn = pd.read_csv('../input/test_transaction.csv')



print(train_id.shape, test_id.shape)

print(train_trn.shape, test_trn.shape)
id_cols = list(train_id.columns.values)

trn_cols = list(train_trn.drop('isFraud', axis=1).columns.values)



X_train = pd.merge(train_trn[trn_cols + ['isFraud']], train_id[id_cols], how='left')

X_train = reduce_mem_usage(X_train)

X_test = pd.merge(test_trn[trn_cols], test_id[id_cols], how='left')

X_test = reduce_mem_usage(X_test)



X_train_id = X_train.pop('TransactionID')

X_test_id = X_test.pop('TransactionID')

del train_id,train_trn,test_id,test_trn



all_data = X_train.append(X_test, sort=False).reset_index(drop=True)
_='''

corr_matrix = all_data.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

columns_to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]

del upper



print('drop columns:', columns_to_drop)

all_data.drop(columns_to_drop, axis=1, inplace=True)

'''
def encode_loop(df, col, drop=True):

    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())

    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())

    if drop:

        df.drop(col, axis=1, inplace=True)

    return df
ccols = [f'C{i}' for i in range(1,15)]

dcols = [f'D{i}' for i in range(1,16)]

mcols = ['M1','M2','M3','M5','M6','M7','M8','M9']
all_data['_log_dist_1_2'] = np.log1p(np.where(all_data['dist1'].isna(), all_data['dist2'], all_data['dist1']))
import datetime



START_DATE = '2017-11-30'

#START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

trandate = all_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))



all_data['_days'] = all_data['TransactionDT'] // (24*60*60)

#all_data['_weekday'] = trandate.dt.dayofweek.astype(str)

all_data['_hour'] = trandate.dt.hour

all_data = encode_loop(all_data, '_hour')

#all_data['_day'] = trandate.dt.day

#all_data['_year_month'] = trandate.dt.year.astype(str) + '_' + trandate.dt.month.astype(str)

all_data['_weekday__hour'] = trandate.dt.dayofweek.astype(str) + '_' + trandate.dt.hour.astype(str)
all_data['_P_emaildomain__ProductCD'] = all_data['P_emaildomain'] + '__' + all_data['ProductCD']

all_data['_card3__card5'] = all_data['card3'].astype(str) + '__' + all_data['card5'].astype(str)
all_data['_uid1'] = (all_data['_days'] - all_data['D1']).astype(str) + '__' + all_data['P_emaildomain'].astype(str)

all_data['_uid2'] = all_data['card1'].astype(str) + '__' + all_data['addr1'].astype(str) + '__' + all_data['_uid1']



# lag previous transaction

group_key = ['_uid2']

all_data = all_data.assign(

        _day_lag_uid2 = all_data['TransactionDT'] - all_data.groupby(group_key)['TransactionDT'].shift(1)

        #,_amount_lag_uid2 = all_data['TransactionAmt'] - all_data.groupby(group_key)['TransactionAmt'].shift(1)

        #,_amount_ema5_uid2 = all_data.groupby(group_key)['TransactionAmt'].apply(lambda x: x.ewm(span=5).mean())

        ,_amount_lag_pct_uid2 = np.abs(all_data.groupby(group_key)['TransactionAmt'].pct_change())

)

_='''

group_key = ['_uid1']

all_data = all_data.assign(

        _day_lag_uid1 = all_data['TransactionDT'] - all_data.groupby(group_key)['TransactionDT'].shift(1)

        ,_amount_lag_uid1 = all_data['TransactionAmt'] - all_data.groupby(group_key)['TransactionAmt'].shift(1)

        ,_amount_ema5_uid1 = all_data.groupby(group_key)['TransactionAmt'].apply(lambda x: x.ewm(span=5).mean())

        ,_amount_lag_pct_uid1 = np.abs(all_data.groupby(group_key)['TransactionAmt'].pct_change())

)

'''
all_data['_amount_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)

all_data['_amount_decimal_len'] = all_data['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))

all_data['_amount_fraction'] = all_data['TransactionAmt'].apply(lambda x: float('0.'+re.sub('^[0-9]|\.|0+$', '', str(x))))
def values_agg(df, periods, columns, aggs=['max']):

    for period in periods:

        for col in columns:

            if col in df.columns:

                new_col = f'{col}_{period}'

                grouped_col = df.groupby([period])[col]

                for a in aggs:

                    df[f'_{a}_{new_col}'] = df[period].map(grouped_col.agg(a).to_dict())

    return df



all_data = values_agg(all_data, ['_uid2'], ccols)

#all_data = values_agg(all_data, ['_uid2'], dcols)
amt_cols = ['_uid2','_P_emaildomain__ProductCD']



all_data = values_agg(all_data, amt_cols, ['TransactionAmt'], aggs=['max','mean','var'])
for f in ccols + amt_cols:

    vc = all_data[f].value_counts(dropna=False)

    all_data[f'_count_full_{f}'] = all_data[f].map(vc)
all_data['_all_na'] = all_data.isna().sum(axis=1).astype(np.int8)

#all_data['_addr_na'] = all_data[['addr1','addr2','dist1','dist2']].isna().sum(axis=1).astype(np.int8)
all_data['_ccols_nonzero'] = all_data[ccols].apply(lambda x: len(x.to_numpy().nonzero()[0]), axis=1)

all_data['_ccols_sum'] = all_data[ccols].sum(axis=1).astype(np.int8)

all_data['_ccols_0_bin'] = ''

for c in ccols:

    all_data['_ccols_0_bin'] += (all_data[c] == 0).astype(int).astype(str)



all_data.drop(ccols, axis=1, inplace=True)
#all_data['_D1_eq_D2'] = np.where(all_data['D1'] == all_data['D2'].fillna(0),'1','0')

#all_data['_D3_eq_D5'] = np.where(all_data['D3'] == all_data['D5'],'1','0')

#all_data['_D8_na'] = np.where(np.isnan(all_data['D8']),'1','0')



all_data['_dcol_na'] = all_data[dcols].isna().sum(axis=1).astype(np.int8)

all_data['_dcols_na_bin'] = ''

for c in dcols:

    all_data['_dcols_na_bin'] += all_data[c].isna().astype(int).astype(str)



# diff date threshold

for f in ['D1','D2']:

    #all_data[f] = all_data[f].fillna(0) - all_data['_days'].apply(lambda x: np.min([154,x]))

    all_data[f] = all_data[f].fillna(0) - all_data['_days']



for f in ['D3','D4','D5','D6','D7','D10','D11','D12','D13','D14','D15']:

    all_data[f] = all_data[f].fillna(0) - all_data['_days']



#all_data['_dcol_max'] = all_data[dcols].fillna(0).max(axis=1).astype(np.int8)

    

# time feature

#all_data['D9'] = (all_data['D9'] * 24)

#all_data['_D9_na'] = all_data['D9'].isna().astype(np.int8)



#all_data.drop(dcols, axis=1, inplace=True)
#all_data['_mcol_sum'] = all_data[mcols].sum(axis=1).astype(np.int8)

#all_data['_mcol_na'] = all_data[mcols].isna().sum(axis=1).astype(np.int8)

all_data['_mcols_na_bin'] = ''

for c in mcols:

    all_data['_mcols_na_bin'] += all_data[c].isna().astype(int).astype(str)
vcols = [f'V{i}' for i in range(1,340)]



sc = preprocessing.MinMaxScaler()



dec = PCA(n_components=2, random_state=42) #0.99

vcol_dec = dec.fit_transform(sc.fit_transform(all_data[vcols].fillna(-1)))



all_data['_vcols_dec0'] = vcol_dec[:,0]

all_data['_vcols_dec1'] = vcol_dec[:,1]

all_data['_vcols_na'] = all_data[vcols].isna().sum(axis=1).astype(np.int8)



for f in ['V144','V145','V150','V151','V159','V160','V307']:

    vcols.remove(f)



all_data['_vcols_sum'] = all_data[vcols].sum(axis=1).astype(np.int8)



all_data.drop(vcols, axis=1, inplace=True)
_='''

cnt_day = cnt_day / cnt_day.mean()

all_data['_pct_trns_day'] = all_data['_days'].map(cnt_day.to_dict())

'''

cnt_day = all_data['_days'].value_counts()

all_data['_count_trns_day'] = all_data['_days'].map(cnt_day.to_dict())



#daily_cols = ['C1','C13'] # + amt_cols + dcols

daily_cols = ccols

for f in daily_cols:

    if f in all_data.columns:

        val_day = all_data[f].astype(str) + '__' + all_data['_days'].astype(str)

        vc = val_day.value_counts(dropna=True)

        all_data[f'_count_day_{f}'] = val_day.map(vc)

        all_data[f'_count_pct_day_{f}'] = all_data[f'_count_day_{f}'] / all_data['_count_trns_day']



all_data.drop('_count_trns_day', axis=1, inplace=True)
# drop low impotance features

_='''

'''

columns_to_drop = ['id_07','id_08','id_10','id_12','id_16',

                   'id_21','id_22','id_23','id_24','id_25',

                   'id_26','id_27','id_28','id_29','id_32',

                   'id_34','id_35','id_36','id_37']

all_data.drop(columns_to_drop, axis=1, inplace=True)
many_same_values_columns = [c for c in all_data.drop('isFraud', axis=1).columns if all_data[c].value_counts(normalize=True).values[0] >= 0.98]

columns_to_drop = list(many_same_values_columns)

print('drop columns:', columns_to_drop)

all_data.drop(columns_to_drop, axis=1, inplace=True)
# drop feature

all_data.drop(['_days'], axis=1, inplace=True)

all_data.drop(['TransactionDT'], axis=1, inplace=True) #,'card3','card5'



uid = all_data['_uid2']



drop_card_cols = ['_uid1','_uid2','card1','card2']

#all_data.drop(drop_card_cols, axis=1, inplace=True)
_='''

'''

cat_cols = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain',

            'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo'] + [f'id_{i}' for i in range(12,39)]



# to str type

for i in cat_cols:

    if i in all_data.columns:

        all_data[i] = all_data[i].astype(str)

        #all_data[i].fillna('unknown', inplace=True) # need for category-type



# factorize

enc_cols = []

for i, t in all_data.loc[:, all_data.columns != 'isFraud'].dtypes.iteritems():

    if t == object:

        enc_cols.append(i)

        all_data[i] = pd.factorize(all_data[i])[0]

        #all_data[i] = all_data[i].astype('category')

print(enc_cols)
print('features:', all_data.shape[1])
X_train = all_data[all_data['isFraud'].notnull()]

X_test = all_data[all_data['isFraud'].isnull()].drop('isFraud', axis=1)

Y_train = X_train.pop('isFraud')



uid_train = uid[all_data['isFraud'].notnull()]

uid_fraud = uid_train[Y_train == 1]

uid_test = uid[all_data['isFraud'].isnull()]



train_group = trandate[:len(X_train)].dt.month



del uid

del all_data
_='''

pseudo = X_test[uid_test.isin(uid_fraud).values]

pseudo['isFraud'] = 1



Y_train = pd.concat([Y_train, pseudo.pop('isFraud')], axis=0)

X_train = pd.concat([X_train, pseudo], axis=0)

p_group = pd.Series([train_group[0]] * len(pseudo))

train_group = pd.concat([train_group, p_group], axis=0)

print(len(pseudo))

'''
cat_cols = ['card1','card2','card3','card5','addr1','_uid1','_uid2','_P_emaildomain__ProductCD','_card3__card5']

for f in cat_cols:

    if f in X_train.columns:

        train_set = set(X_train[f])

        test_set = set(X_test[f])

        tt = train_set.intersection(test_set)

        print(f, '-', 'train:%.3f'%(len(tt)/len(train_set)), ',test:%.3f'%(len(tt)/len(test_set)))

        X_train[f] = X_train[f].map(lambda x: -999 if x not in tt else x)

        X_test[f] = X_test[f].map(lambda x: -999 if x not in tt else x)
import lightgbm as lgb

from imblearn.datasets import make_imbalance



def scale_minmax(preds):

    return (preds - preds.min()) / (preds.max() - preds.min())



def pred_lgb(X_train, Y_train, X_test, params, num_iterations=1000):

    feature_importances = pd.DataFrame()

    feature_importances['feature'] = X_train.columns

 

    oof_preds = np.zeros(X_train.shape[0])

    sub_preds = np.zeros(X_test.shape[0])



    dtrain = lgb.Dataset(X_train[:312574], label=Y_train.iloc[:312574])

    dvalid = lgb.Dataset(X_train[-175998:], label=Y_train[-175998:])



    clf = lgb.train(params, dtrain, 5000, 

                    valid_sets=[dtrain, dvalid], valid_names=['train', 'valid'],

                    verbose_eval=1000, early_stopping_rounds=200)

    feature_importances['fold_0'] = clf.feature_importance()

    oof_preds = clf.predict(X_train)

    sub_preds = clf.predict(X_test)

    

    return oof_preds, sub_preds, feature_importances



# predict with lightgbm, KFold 

def pred_lgb_kfold(X_train, Y_train, X_test, params, nfolds=5):

    feature_importances = pd.DataFrame()

    feature_importances['feature'] = X_train.columns



    oof_preds = np.zeros(X_train.shape[0])

    sub_preds = np.zeros(X_test.shape[0])



    kf = KFold(n_splits=nfolds, shuffle=False, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(X_train, Y_train)):

        dtrain = lgb.Dataset(X_train.iloc[train_index], label=Y_train.iloc[train_index])

        dvalid = lgb.Dataset(X_train.iloc[test_index], label=Y_train.iloc[test_index])



        clf = lgb.train(params, dtrain, 5000, 

                        valid_sets=[dtrain, dvalid], valid_names=['train', 'valid'],

                        verbose_eval=1000, early_stopping_rounds=200)

        feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()

        oof_preds[test_index] = clf.predict(X_train.iloc[test_index])

        sub_preds += clf.predict(X_test) / nfolds

    

    return oof_preds, sub_preds, feature_importances
# predict with lightgbm, LeaveOneGroupOut 

def pred_lgb_LOGO(X_train, Y_train, X_test, groups, params, undersampling=False):

    feature_importances = pd.DataFrame()

    feature_importances['feature'] = X_train.columns



    oof_preds = np.zeros(X_train.shape[0])

    sub_preds = np.zeros(X_test.shape[0])



    nfolds = groups.nunique()



    kf = LeaveOneGroupOut()

    for fold, (train_index, test_index) in enumerate(kf.split(X_train, Y_train, groups)):

        if undersampling:

            size1 = sum(Y_train.iloc[train_index]==1)

            X_train_us, Y_train_us = make_imbalance(

                    X_train.iloc[train_index].fillna(-9999), Y_train.iloc[train_index], 

                    sampling_strategy={0:size1*5, 1:size1}, random_state=42)

            X_train_us = pd.DataFrame(X_train_us, columns=X_train.columns)

            dtrain = lgb.Dataset(X_train_us, label=Y_train_us)

        else:

            dtrain = lgb.Dataset(X_train.iloc[train_index], label=Y_train.iloc[train_index])



        dvalid = lgb.Dataset(X_train.iloc[test_index], label=Y_train.iloc[test_index])



        clf = lgb.train(params, dtrain, 5000, 

                        valid_sets=[dtrain, dvalid], valid_names=['train', 'valid'],

                        verbose_eval=1000, early_stopping_rounds=200)

        feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()

        oof_preds[test_index] = clf.predict(X_train.iloc[test_index])

        sub_preds += clf.predict(X_test) / nfolds

    

    return oof_preds, sub_preds, feature_importances
# predict with lightgbm, LeaveOneGroupOut 

def pred_lgb_LOGO_2(X_train, Y_train, X_test, groups, params):

    feature_importances = pd.DataFrame()

    feature_importances['feature'] = X_train.columns



    oof_preds = np.zeros(X_train.shape[0])

    sub_preds = np.zeros(X_test.shape[0])



    nfolds = groups.nunique()

    print('groups:', nfolds)



    kf = LeaveOneGroupOut()

    pred_count = 0

    for fold, (tt_index, out_index) in enumerate(kf.split(X_train, Y_train, groups)):

        train_index = tt_index[tt_index < out_index[0]]

        test_index = tt_index[tt_index > out_index[len(out_index) - 1]]

        print(f'fold[{fold+1}] - {len(tt_index)} (train:{len(train_index)}, valid:{len(test_index)})')

        if len(train_index) == 0 or len(test_index) == 0: # or len(train_index) < len(test_index)

            print('- skip')

            continue

        

        pred_count += 1

        

        dtrain = lgb.Dataset(X_train.iloc[train_index], label=Y_train.iloc[train_index])

        dvalid = lgb.Dataset(X_train.iloc[test_index], label=Y_train.iloc[test_index])



        clf = lgb.train(params, dtrain, 5000, 

                        valid_sets=[dtrain, dvalid], valid_names=['train', 'valid'],

                        verbose_eval=500, early_stopping_rounds=200)

        feature_importances['fold_{}'.format(pred_count)] = clf.feature_importance()

        oof_preds += clf.predict(X_train)

        sub_preds += clf.predict(X_test) 



    oof_preds = oof_preds / pred_count

    sub_preds = sub_preds / pred_count 

        

    return oof_preds, sub_preds, feature_importances
%%time



params={

        #'boosting':'dart', # dart (drop out trees)

        'learning_rate': 0.01,

        'objective': 'binary',

        'boost_from_average': False,

        'is_unbalance': False,

        'metric': 'auc',

        'num_threads': -1,

        'num_leaves': 256,

        'max_bin': 256,

        'verbose': 1,

        'random_state': 42,

        'bagging_fraction': 0.85,

        'bagging_freq': 1,

        'feature_fraction': 0.60

    }



#oof_preds, sub_preds, feature_importances = pred_lgb(X_train, Y_train, X_test, params)

#oof_preds, sub_preds, feature_importances = pred_lgb_kfold(X_train, Y_train, X_test, params, nfolds=3)

oof_preds, sub_preds, feature_importances = pred_lgb_LOGO(X_train, Y_train, X_test, train_group, params) #, undersampling=True

#oof_preds, sub_preds, feature_importances = pred_lgb_LOGO_2(X_train, Y_train, X_test, train_group, params)
fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
# Add pseudo labeled data

_='''

#X_test_p1 = X_test[(sub_preds <= 0.00001)].copy()

#X_test_p1['isFraud'] = 0

#print(X_test_p1.shape)

#Y_train = pd.concat([Y_train, X_test_p1.pop('isFraud')], axis=0)

#X_train = pd.concat([X_train, X_test_p1], axis=0)



X_test_p2 = X_test[(sub_preds >= 0.99)].copy()

X_test_p2['isFraud'] = 1

print(X_test_p2.shape)

Y_train = pd.concat([Y_train, X_test_p2.pop('isFraud')], axis=0)

X_train = pd.concat([X_train, X_test_p2], axis=0)



Y_train.reset_index(drop=True, inplace=True)

X_train.reset_index(drop=True, inplace=True)



oof_preds, sub_preds, feature_importances = pred_lgb_kfold(X_train, Y_train, X_test, params, nfolds=3)



fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)

'''
folds = [c for c in feature_importances.columns if c.startswith('fold')]

feature_importances['average'] = feature_importances[folds].mean(axis=1)

feature_importances.sort_values(by='average', ascending=False, inplace=True)

#feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(13, 13))

sns.barplot(data=feature_importances.head(50), x='average', y='feature');

plt.title('50 TOP feature importance');
feature_importances.sort_values(by='average', ascending=False)['feature'].values
submission = pd.DataFrame()

submission['TransactionID'] = X_test_id

submission['isFraud'] = sub_preds



submission.loc[uid_test.isin(uid_fraud).values, 'isFraud'] = 1



submission.to_csv('submission.csv', index=False)
np.mean(sub_preds)