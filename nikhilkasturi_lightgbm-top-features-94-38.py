import pandas as pd

import numpy as np

from tqdm import tqdm_notebook

from sklearn.metrics import roc_auc_score

import gc



from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
folder_path = '../input/'

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
def id_split(dataframe):

    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]

    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]



    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]

    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]



    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]

    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]



    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]

    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]



    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]

    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]



    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    dataframe['had_id'] = 1

    gc.collect()

    

    return dataframe
train_identity = id_split(train_identity)

test_identity = id_split(test_identity)
print('Merging data...')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print('Data was successfully merged!\n')



del train_identity, train_transaction, test_identity, test_transaction



print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')

print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')



gc.collect()
useful_features = ['card1_count_full',

'addr1__card1',

'card1',

'TransactionAmt_to_std_card1',

'card1__card5',

'TransactionAmt_to_mean_card1',

'card2_count_full',

'card2',

'card2__dist1',

'Transaction_hour',

'TransactionAmt_to_std_card4',

'addr1',

'TransactionAmt_to_mean_card4',

'TransactionAmt',

'P_emaildomain__C2',

'card5__P_emaildomain',

'dist1',

'card2__id_20',

'D4',

'Transaction_day_of_week',

'D2',

'D15_to_mean_card1',

'D10',

'TransactionAmt_Log',

'D11',

'D5',

'TransactionAmt_decimal',

'C13',

'D15_to_std_card1',

'D3',

'D1',

'D15_to_mean_addr1',

'DeviceInfo__P_emaildomain',

'D15_to_mean_card4',

'D15',

'card5',

'P_emaildomain',

'D11__DeviceInfo',

'D15_to_std_addr1',

'card5_count_full',

'D15_to_std_card4',

'id_02_to_std_card1',

'V307',

'P_emaildomain_bin',

'id_19',

'id_20',

'id_02_to_mean_card1',

'D8',

'id_02__id_20',

'V310',

'C2',

'V127',

'C1',

'id_02_to_std_card4',

'id_02_to_mean_card4',

'id_02',

'C14',

'C9',

'id_31',

'id_31_count_dist',

'V308',

'C6',

'V130',

'V314',

'id_13',

'version_id_31',

'V313',

'id_02__D8',

'C11',

'V315',

'D9',

'id_06',

'M4',

'V128',

'V312',

'R_emaildomain',

'id_05',

'V306',

'DeviceInfo',

'C5',

'V317',

'M6',

'M5',

'R_emaildomain_bin',

'V285',

'V96',

'V131',

'V126',

'id_01',

'screen_height',

'D6',

'V35',

'V5',

'V36',

'M8',

'device_version',

'V283',

'id_33_count_dist',

'D14',

'card4',

'version_id_30',

'V99',

'id_33',

'V4',

'D12',

'M7',

'device_name',

'V282',

'id_01_count_dist',

'M9',

'V62',

'id_14',

'id_30',

'V83',

'V75',

'V38',

'V264',

'V13',

'V76',

'V78',

'V45',

'V54',

'V53',

'P_emaildomain_suffix',

'V20',

'V87',

'V82',

'M3',

'V280',

'D13',

'card4_count_full',

'V291',

'V12',

'V61',

'V294',

'card6',

'V44',

'V19',

'C10',

'C12',

'V37',

'V56',

'V49',

'V11',

'V265',

'screen_width',

'V48',

'V203',

'V10',

'C8',

'V3',

'V7',

'card3',

'M2',

'browser_id_31',

'V97',

'V221',

'card6_count_full',

'V30',

'V165',

'ProductCD',

'V263',

'V29',

'V279',

'card3_count_full',

'V70',

'V91',

'V267',

'V69',

'V204',

'V90',

'id_38',

'V160',

'V150',

'V6',

'V222',

'V52',

'V143',

'V292',

'V202',

'V274',

'V287',

'V289',

'V159',

'V40',

'id_09',

'V164',

'V145',

'V332',

'V166',

'V51',

'V139',

'V95',

'C4',

'V209',

'V47',

'V277',

'id_03',

'V74',

'V208',

'V261',

'V207',

'V266',

'V212',

'V268',

'V275',

'V273',

'V229',

'V258',

'V9',

'V333',

'DeviceType',

'V210',

'id_11',

'V100',

'id_17',

'V331',

'V140',

'V219',

'V262',

'V215',

'V213',

'V278',

'V206',

'V270',

'V271',

'V8',

'V234',

'R_emaildomain_suffix',

'V152',

'V276',

'V151',

'V323',

'id_15',

'V217',

'V170',

'V214',

'V171',

'V205',

'V245',

'C7',

'V73',

'id_37',

'V81',

'V34',

'V288',

'V272',

'V216',

'V224',

'V338',

'id_32',

'V46',

'V259',

'V228',

'V169',

'V149',

'V64',

'V58',

'OS_id_30',

'V178',

'V257',

'V187',

'V60',

'V33',

'V233',

'V80',

'V94',

'V220',

'V251',

'V59',

'V167',

'V226',

'V324',

'V85',

'V176',

'V156',

'V243',

'V335',

'V231',

'V201',

'V189',

'V180',

'addr2',

'V71',

'V63',

'V72',

'V177',

'V225',

'V322',

'V17',

'V256',

'V246',

'V93',

'V200',

'V182',

'V329',

'V184',

'V303',

'V84',

'V92',

'V336',

'V244',

'V158',

'V238',

'V223',

'V326',

'V173',

'V172',

'V147',

'V188',

'V239',

'V253',

'V175',

'V242',

'V146',

'id_36_count_full',

'V154',

'V227',

'V304',

'V162',

'V161',

'id_12',

'V249',

'V163',

'id_36_count_dist',

'id_36',

'V195',

'V197',

'V247',

'V138',

'had_id'



]

cols_to_drop = [col for col in train.columns if col not in useful_features]

cols_to_drop.remove('isFraud')

cols_to_drop.remove('TransactionDT')
train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')



train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')



train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')
# New feature - log of transaction amount. ()

train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])

test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])



# New feature - decimal part of the transaction amount.

train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# New feature - day of week in which a transaction happened.

train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)



# New feature - hour of the day in which a transaction happened.

train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24



# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))



# Encoding - count encoding for both train and test

for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']:

    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))



# Encoding - count encoding separately for train and test

for feature in ['id_01', 'id_31', 'id_33']:

    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']
for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin'] = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
for col in train.columns:

    if train[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values))
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.head()
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)

y = train.sort_values('TransactionDT')['isFraud']



X_test = test.drop(['TransactionDT'], axis=1)



del train, test

gc.collect()
from sklearn.model_selection import KFold

import lightgbm as lgb
params = {'num_leaves': 951,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 280,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.020883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'max_delta_step' : 1,

          'is_unbalance': 'True',

          'reg_lambda': 0.6485237330340494,

          'random_state': 47,

           'n_jobs':-1,

           'tree_learner':'serial',

          'colsample_bytree': 0.7,

          'subsample_freq':1,

          'subsample':0.7,

         }





NFOLDS = 15

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 20000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(X_test) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
sample_submission['isFraud'] = y_preds

sample_submission.to_csv("submission.csv", index=False)
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances2.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));