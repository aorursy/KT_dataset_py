import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time

import multiprocessing

import warnings

warnings.filterwarnings(action='ignore')

%%time



Df_TrainID = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

Df_TrainTransaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

Df_TestID = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

Df_TestTransaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

#Df_Sample = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

   
Df_TrainID.head()
Df_TrainTransaction.head()
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
train_id = reduce_mem_usage(Df_TrainID)

train_trn = reduce_mem_usage(Df_TrainTransaction)

test_id = reduce_mem_usage(Df_TestID)

test_trn = reduce_mem_usage(Df_TestTransaction)
id_cols = list(train_id.columns.values)

trn_cols = list(train_trn.drop('isFraud', axis=1).columns.values)



X_train = pd.merge(train_trn[trn_cols + ['isFraud']], train_id[id_cols], how='left')

X_test = pd.merge(test_trn[trn_cols], test_id[id_cols], how='left')



X_train_id = X_train.pop('TransactionID')

X_test_id = X_test.pop('TransactionID')

del train_id,train_trn,test_id,test_trn



all_data = X_train.append(X_test, sort=False).reset_index(drop=True)
X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
from sklearn import metrics, preprocessing

from sklearn.decomposition import PCA,KernelPCA



sc = preprocessing.MinMaxScaler()
vValues = [f'V{i}' for i in range(1,340)]

pca = PCA(n_components=2)

vPCA = pca.fit_transform(sc.fit_transform(all_data[vValues].fillna(-1)))



all_data['_vcol_pca0'] = vPCA[:,0]

all_data['_vcol_pca1'] = vPCA[:,1]

all_data['_vcol_nulls'] = all_data[vValues].isnull().sum(axis=1)



all_data.drop(vValues, axis=1, inplace=True)
import datetime
START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

all_data['Date'] = all_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

all_data['_weekday'] = all_data['Date'].dt.dayofweek

all_data['_hour'] = all_data['Date'].dt.hour

all_data['_day'] = all_data['Date'].dt.day



all_data['_weekday'] = all_data['_weekday'].astype(str)

all_data['_hour'] = all_data['_hour'].astype(str)

all_data['_weekday__hour'] = all_data['_weekday'] + all_data['_hour']



cnt_day = all_data['_day'].value_counts()

cnt_day = cnt_day / cnt_day.mean()

all_data['_count_rate'] = all_data['_day'].map(cnt_day.to_dict())



all_data.drop(['TransactionDT','Date','_day'], axis=1, inplace=True)
all_data['_P_emaildomain__addr1'] = all_data['P_emaildomain'] + '__' + all_data['addr1'].astype(str)

all_data['_card1__card2'] = all_data['card1'].astype(str) + '__' + all_data['card2'].astype(str)

all_data['_card1__addr1'] = all_data['card1'].astype(str) + '__' + all_data['addr1'].astype(str)

all_data['_card2__addr1'] = all_data['card2'].astype(str) + '__' + all_data['addr1'].astype(str)

all_data['_card12__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)

all_data['_card_all__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)
import re
all_data['_amount_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)

all_data['_amount_decimal_len'] = all_data['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))

all_data['_amount_fraction'] = all_data['TransactionAmt'].apply(lambda x: float('0.'+re.sub('^[0-9]|\.|0+$', '', str(x))))

all_data[['TransactionAmt','_amount_decimal','_amount_decimal_len','_amount_fraction']].head(10)
cols = ['ProductCD','card1','card2','card5','card6','P_emaildomain','_card_all__addr1']

for f in cols:

    all_data[f'_amount_mean_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('mean')

    all_data[f'_amount_std_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('std')

    all_data[f'_amount_pct_{f}'] = (all_data['TransactionAmt'] - all_data[f'_amount_mean_{f}']) / all_data[f'_amount_std_{f}']



for f in cols:

    vc = all_data[f].value_counts(dropna=False)

    all_data[f'_count_{f}'] = all_data[f].map(vc)
cat_cols = [f'id_{i}' for i in range(12,39)]

for i in cat_cols:

    if i in all_data.columns:

        all_data[i] = all_data[i].astype(str)

        all_data[i].fillna('unknown', inplace=True)



enc_cols = []

for i, t in all_data.loc[:, all_data.columns != 'isFraud'].dtypes.iteritems():

    if t == object:

        enc_cols.append(i)

        all_data[i] = pd.factorize(all_data[i])[0]

print(enc_cols)
X_train = all_data[all_data['isFraud'].notnull()]

X_test = all_data[all_data['isFraud'].isnull()].drop('isFraud', axis=1)

Y_train = X_train.pop('isFraud')

del all_data
%%time



import lightgbm as lgb
params={'learning_rate': 0.01,

        'objective': 'binary',

        'metric': 'auc',

        'num_threads': -1,

        'num_leaves': 256,

        'verbose': 2,

        'random_state': 42,

        'bagging_fraction': 1,

        'feature_fraction': 0.88

       }



oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])



clf = lgb.LGBMClassifier(**params, n_estimators=4000)

clf.fit(X_train, Y_train)

oof_preds = clf.predict_proba(X_train, num_iteration=clf.best_iteration_)[:,1]

sub_preds = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:,1]
Final = pd.DataFrame()

Final['TransactionID'] = X_test_id

Final['isFraud'] = sub_preds
Final.to_csv('Final.csv', index=False)
import matplotlib.pyplot as plt

false, true, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(false, true)



plt.plot(false, true, label='ROC curve (area = %.3f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)