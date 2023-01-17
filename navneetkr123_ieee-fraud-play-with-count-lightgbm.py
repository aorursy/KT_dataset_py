imp_feature = ['uid3_count_dist', 'feat1', 'TransactionAmt', 'feat2', 'card1_count_dist', 'card2_count_dist', 'addr1_count_dist', 'uid_count_dist', 'uid2_count_dist', 'Transaction_hour', 'uid1_count_dist', 'P_emaildomain_count_dist', 'M_na', 'M_na_count_dist', 'D15_count_dist', 'Transaction_day_of_week', 'D1', 'card5_count_dist', 'dist1_count_dist', 'C13', 'D10_count_dist', 'id_20_count_dist', 'D4_count_dist', 'id_19_count_dist', 'C13_count_dist', 'D2_count_dist', 'id_31_count_dist', 'D1_count_dist', 'DeviceInfo_count_dist', 'R_emaildomain_count_dist', 'C1', 'uid4_count_dist', 'id_33_count_dist', 'V310_count_dist', 'C2', 'C1_count_dist', 'V307_count_dist', 'C14', 'C6', 'C2_count_dist', 'C14_count_dist', 'C6_count_dist', 'id_30_count_dist', 'D3_count_dist', 'V313_count_dist', 'D11_count_dist', 'D5_count_dist', 'id_05_count_dist', 'D9_count_dist', 'C11', 'dist2_count_dist', 'C9', 'V315_count_dist', 'id_13_count_dist', 'C9_count_dist', 'id_01_count_dist', 'V130_count_dist', 'M4_count_dist', 'C11_count_dist', 'V314_count_dist', 'M6_count_dist', 'M5_count_dist', 'card4_count_dist', 'V127_count_dist', 'V312_count_dist', 'D14_count_dist', 'V308_count_dist', 'C5', 'V283_count_dist', 'id_14_count_dist', 'V83_count_dist', 'M3_count_dist', 'C5_count_dist', 'V62_count_dist', 'id_18_count_dist', 'ProductCD_count_dist', 'V87_count_dist', 'V282_count_dist', 'V317_count_dist', 'D6_count_dist', 'card6_count_dist', 'V285_count_dist', 'V76_count_dist', 'card3_count_dist', 'id_02_count_dist', 'V45_count_dist', 'V143_count_dist', 'V38_count_dist', 'V131_count_dist', 'C10']

import pandas as pd

import numpy as np

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
sub = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")
sub.shape
train_id = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

train_tr = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
train_id.head(5)
train_tr.head(5)
train_id.shape, train_tr.shape
test_id = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

test_tr = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
test_id.shape, test_tr.shape
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

test = pd.merge(test_tr, test_id, on='TransactionID', how='left')



del test_id, test_tr

del train_id, train_tr

gc.collect()
train.head(5)
# Negative downsampling

train_pos = train[train['isFraud']==1]

train_neg = train[train['isFraud']==0]



train_neg = train_neg.sample(2*int(train_pos.shape[0] ), random_state=42)

train = pd.concat([train_pos,train_neg]).sort_index()
train_pos.shape, train_neg.shape
l = 3*int(train_pos.shape[0])
train.shape
#train.head(2)
del train_pos

del train_neg
train.isna().sum()
train = train.sort_values('TransactionDT')
train.shape
train.drop_duplicates(inplace=True)
train.shape
#
#corr = train.corr()

#corr.head(403)

#Correlation with output variable

#cor_target = abs(corr["isFraud"])

#Selecting highly correlated features

#relevant_features = cor_target[cor_target>0.05]

#relevant_features
target = train["isFraud"]

train.drop(["isFraud"], axis=1, inplace=True)
useful_features = [col for col in train.columns]
len(useful_features)
train.shape
train.isna().sum()
pd.set_option('display.max_columns', None)
train.shape, test.shape
train.head(10)
#target = train["isFraud"]

#train.drop(["isFraud"], axis=1, inplace=True)
train = train.iloc[:, 1:31]

test = test.iloc[:, 1:31]
train.head()
train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24

train['uid'] = train['card1'].astype(str)+'_'+train['card2'].astype(str)

test['uid'] = test['card1'].astype(str)+'_'+test['card2'].astype(str)



train['uid1'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)

test['uid1'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)





train['uid2'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)

test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)



train['uid3'] = train['uid2'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)

test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)



train['uid4'] = train['card4'].astype(str)+'_'+train['card6'].astype(str)

test['uid4'] = test['card4'].astype(str)+'_'+test['card6'].astype(str)



train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)

test['TransactionAmt_check']  = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)

train = pd.concat([train, test])
#neglect = ["TransactionAmt", 'Transaction_day_of_week', 'Transaction_hour']
#useful_features = [col for col in train.columns if col not in neglect]
train['M_na'] = abs(train.isna().sum(axis=1).astype(np.int8))
#non_nan = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",\

 #         "D1", "M_na"]
#"V96", "V97", "V98", "V99", "V100", "V101", "V102", "V103", "V104", "V105", "V106",\

#          "V107", "V108", "V109", "V110", "V111", "V112", "V113", "V114", "V115", "V116", "V117","V118",\

 #         "V119", "V120", "V121","V122","V123", "V124", "V125", "V126", "V127", "V128", "V129", "V130",\

  #        "V131", "V132", "V133", "V134", "V135", "V136", "V137", "V297", "V298", "V299", "V300",\

   #       "V301", "V301", "V302", "V303", "V304", "V305", "V306", "V307", "V308", "V309", "V310",\

    #      "V311", "V312", "V313", "V314", "V315", "V316", "V317", "V318", "V319", "V320", "V321",\

     #     "V279", "V280", "V281", "V282", "V283", "V284", "V285", "V286", "V287", "V288", "V289", "V290",\

      #    "V291", "V292", "V293", "V294", "V295", "V296"]
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

    

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = 'Others'

    gc.collect()

    return dataframe
#train = id_split(train)
#train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])

train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
neglect = ["TransactionAmt", 'Transaction_day_of_week', 'Transaction_hour', "TransactionAmt_Log", 'TransactionAmt_decimal']
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 

          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 

          'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 

          'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 

          'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 

          'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 

          'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 

          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 

          'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 

          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 

          'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 

          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 

          'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 

          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 

          'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 

          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 

          'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 

          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']
for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
useful_features = [col for col in train.columns if col not in neglect]
#i=0        

#for feature in useful_features:

    

        # Count encoded separately for train and test

 #   train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

    #if feature not in non_nan:

     #   train.drop([feature], axis=1,inplace=True)

  #  print("Done" + str(i))

   # i+=1

        
dropping =["D8_count_dist", "V138_count_dist", "V139_count_dist", "V140_count_dist", "V141_count_dist",\

           "V146_count_dist", "V147_count_dist", "V148_count_dist", "V149_count_dist", "V144_count_dist",\

           "V145_count_dist", "V150_count_dist", "V151_count_dist", "V152_count_dist", "V153_count_dist",\

           "V154_count_dist", "V155_count_dist", "V156_count_dist", "V157_count_dist", "V158_count_dist",\

           "V159_count_dist", "V160_count_dist", "V161_count_dist", "V162_count_dist", "V163_count_dist",\

           "V164_count_dist", "V165_count_dist", "V166_count_dist", "V168_count_dist", "V170_count_dist",\

           "V171_count_dist", "V172_count_dist", "V173_count_dist", "V174_count_dist", "V175_count_dist",\

           "V176_count_dist", "V177_count_dist", "V178_count_dist", "V179_count_dist", "V180_count_dist",\

           "V181_count_dist", "V182_count_dist", "V183_count_dist", "V184_count_dist", "V185_count_dist",\

           "V186_count_dist", "V187_count_dist", "V188_count_dist", "V189_count_dist", "V190_count_dist",\

           "V191_count_dist", "V192_count_dist", "V193_count_dist", "V194_count_dist", "V195_count_dist",\

           "V196_count_dist", "V197_count_dist", "V198_count_dist", "V199_count_dist", "V200_count_dist",\

           "V201_count_dist", "V202_count_dist", "V203_count_dist", "V204_count_dist", "V205_count_dist",\

           "V206_count_dist", "V207_count_dist", "V208_count_dist", "V209_count_dist", "V210_count_dist",\

           "V211_count_dist", "V212_count_dist", "V213_count_dist", "V214_count_dist", "V215_count_dist",\

           "V216_count_dist", "V218_count_dist", "V219_count_dist", "V221_count_dist", "V222_count_dist",\

           "V223_count_dist", "V224_count_dist", "V225_count_dist", "V226_count_dist", "V227_count_dist",\

           "V228_count_dist", "V229_count_dist", "V230_count_dist", "V231_count_dist", "V232_count_dist",\

           "V233_count_dist", "V234_count_dist", "V235_count_dist", "V236_count_dist", "V237_count_dist",\

           "V205_count_dist", "V205_count_dist", "V205_count_dist", "V205_count_dist", "V205_count_dist",\

           "V238_count_dist", "V239_count_dist", "V240_count_dist", "V241_count_dist", "V242_count_dist",\

           "V243_count_dist", "V244_count_dist", "V245_count_dist","V246_count_dist", "V247_count_dist",\

           "V248_count_dist", "V249_count_dist", "V250_count_dist", "V251_count_dist", "V252_count_dist",\

           "V253_count_dist", "V254_count_dist", "V255_count_dist", "V256_count_dist", "V257_count_dist",\

           "V258_count_dist", "V259_count_dist", "V260_count_dist", "V261_count_dist", "V262_count_dist",\

           "V263_count_dist", "V264_count_dist", "V265_count_dist", "V266_count_dist", "V267_count_dist",\

           "V268_count_dist", "V269_count_dist", "V270_count_dist", "V271_count_dist", "V272_count_dist",\

           "V273_count_dist", "V274_count_dist", "V275_count_dist", "V276_count_dist", "V277_count_dist",\

           "V278_count_dist", "V323_count_dist", "V324_count_dist", "V325_count_dist", "V326_count_dist",\

           "V327_count_dist", "V328_count_dist", "V329_count_dist", "V330_count_dist", "V331_count_dist",\

           "V332_count_dist", "V333_count_dist", "V334_count_dist", "V335_count_dist", "V336_count_dist",\

           "V237_count_dist", "V238_count_dist", "V239_count_dist", "id_04_count_dist", "id_06_count_dist",\

           "id_08_count_dist", "id_10_count_dist", "id_22_count_dist", "id_27_count_dist", "id_29_count_dist",\

           "id_36_count_dist", "id_37_count_dist", "id_38_count_dist"]

#train = train.drop(dropping, axis=1)
#len(dropping)
#train['feat1'] = (train["TransactionAmt"]/train['uid_count_dist'])*100

#train['feat2'] = (train["TransactionAmt"]/train['uid3_count_dist'])*100

#train['feat3'] = (train["TransactionAmt"]/train['card1_count_dist'])*100

#useful_features = [col for col in train.columns]

#for feature in useful_features:

    

        # Count encoded separately for train and test

#    train[feature] = np.log(train[feature])

    

    

    
train.head(4)
train.shape
#X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)#

#y = train.sort_values('TransactionDT')['isFraud']

#test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
train.isna().sum()
train.fillna(-333, inplace=True)
#del train

#gc.collect()
X = train.iloc[:l, :]

test = train.iloc[l:, :]
from sklearn import preprocessing

for f in X.columns:

    if X[f].dtype.name =='object' or test[f].dtype.name =='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X[f].values) + list(test[f].values))

        X[f] = lbl.transform(list(X[f].values))

        test[f] = lbl.transform(list(test[f].values))
#X = train
y=target
X.head()
# Training and Validation Set

#from sklearn.model_selection import train_test_split

#X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=0.20, random_state=23)
from catboost import CatBoostClassifier

categorical_var = np.where(train.dtypes != np.float)[0]

print('\nCategorical Variables indices : ',categorical_var)
del train
#X['feat3'] = (X["TransactionAmt"]/X['card1_count_dist'])*100

#X['feat2'] = (X["TransactionAmt"]/X['uid3_count_dist'])*100



params = {'num_leaves': 491,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47,

          "n_jobs" : -1

         }
folds = TimeSeriesSplit(n_splits=5)



aucs = list()

feature_importances = pd.DataFrame()

feature_importances['feature'] = X.columns



training_start_time = time()

for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

    start_time = time()

    print('Training on fold {}'.format(fold + 1))

    

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)

    

    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()

    aucs.append(clf.best_score['valid_1']['auc'])

    

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

print('-' * 30)

print('Training has finished.')

print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))

print('Mean AUC:', np.mean(aucs))

print('-' * 30)
feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));
# clf right now is the last model, trained with 80% of data and validated with 20%

best_iter = clf.best_iteration
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)

clf.fit(X, y)
sub['isFraud'] = clf.predict_proba(test)[:, 1]
sub.to_csv('ieee_cis_fraud_detection_new.csv', index=False)