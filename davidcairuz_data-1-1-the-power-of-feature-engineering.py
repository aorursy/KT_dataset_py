import numpy as np

import pandas as pd

import os

import gc



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GroupKFold

import xgboost as xgb
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
# COLUMNS WITH STRINGS

str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',

            'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 

            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']



# FIRST 53 COLUMNS

cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',

       'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

       'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',

       'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',

       'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',

       'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',

       'M5', 'M6', 'M7', 'M8', 'M9']



# V COLUMNS TO LOAD DECIDED BY CORRELATION EDA

# https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id

v =  [1, 3, 4, 6, 8, 11]

v += [13, 14, 17, 20, 23, 26, 27, 30]

v += [36, 37, 40, 41, 44, 47, 48]

v += [54, 56, 59, 62, 65, 67, 68, 70]

v += [76, 78, 80, 82, 86, 88, 89, 91]



#v += [96, 98, 99, 104] #relates to groups, no NAN 

v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN

v += [124, 127, 129, 130, 136] # relates to groups, no NAN



# LOTS OF NAN BELOW

v += [138, 139, 142, 147, 156, 162] #b1

v += [165, 160, 166] #b1

v += [178, 176, 173, 182] #b2

v += [187, 203, 205, 207, 215] #b2

v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2

v += [218, 223, 224, 226, 228, 229, 235] #b3

v += [240, 258, 257, 253, 252, 260, 261] #b3

v += [264, 266, 267, 274, 277] #b3

v += [220, 221, 234, 238, 250, 271] #b3



v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN

v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN

v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN

#v += [332, 325, 335, 338] # b4 lots NAN



cols += ['V'+str(x) for x in v]

dtypes = {}

for c in cols+['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,34)]: 

    dtypes[c] = 'float32'

for c in str_type: dtypes[c] = 'object'
%%time



folder_path = '../input/ieee-fraud-detection/'

print('Loading data...')



train_identity = pd.read_csv(f'{folder_path}train_identity.csv', index_col='TransactionID',  dtype=dtypes)

print('\tSuccessfully loaded train_identity!')



train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv', index_col='TransactionID', dtype=dtypes, usecols=cols+['isFraud'])

print('\tSuccessfully loaded train_transaction!')



test_identity = pd.read_csv(f'{folder_path}test_identity.csv', index_col='TransactionID', dtype=dtypes)

print('\tSuccessfully loaded test_identity!')



test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv', index_col='TransactionID', dtype=dtypes, usecols=cols)

print('\tSuccessfully loaded test_transaction!')



sub = pd.read_csv(f'{folder_path}sample_submission.csv')

print('\tSuccessfully loaded sample_submission!')



print('Data was successfully loaded!\n')
print('Merging data...')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print('Data was successfully merged!\n')



del train_identity, train_transaction, test_identity, test_transaction



print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')

print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')



gc.collect()
train.head()
test.head()
%%time



for col in train.columns:

    if train[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values))
import datetime

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')



train['DT_M'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

train['DT_M'] = (train['DT_M'].dt.year-2017)*12 + train['DT_M'].dt.month 



test['DT_M'] = test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

test['DT_M'] = (test['DT_M'].dt.year-2017)*12 + test['DT_M'].dt.month
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)

y = train.sort_values('TransactionDT')['isFraud']



X_test = test.drop(['TransactionDT'], axis=1)



del train, test

gc.collect()
X = X.fillna(-1)

X_test = X_test.fillna(-1)



cols = X.columns
xgb_params = {'n_estimators':5000,

              'max_depth':12,

              'learning_rate':0.02,

              'subsample':0.8,

              'colsample_bytree':0.4,

              'missing':-1,

              'eval_metric':'auc',

              'tree_method':'gpu_hist'

             }
%%time



oof = np.zeros(len(X))

preds = np.zeros(len(X_test))



skf = GroupKFold(n_splits=6)

for i, (idxT, idxV) in enumerate(skf.split(X, y, groups=X['DT_M'])):

    month = X.iloc[idxV]['DT_M'].iloc[0]

    print('Fold',i,'withholding month',month)

    print('rows of train =',len(idxT),'rows of holdout =',len(idxV))

    

    clf = xgb.XGBClassifier(**xgb_params)



    h = clf.fit(X[cols].iloc[idxT], y.iloc[idxT], 

            eval_set=[(X[cols].iloc[idxV],y.iloc[idxV])],

            verbose=100, early_stopping_rounds=200)



    oof[idxV] += clf.predict_proba(X[cols].iloc[idxV])[:,1]

    preds += clf.predict_proba(X_test[cols])[:,1]/skf.n_splits

    

    del h, clf

    x = gc.collect()



print('#'*20)

print ('XGB95 OOF CV=',roc_auc_score(y,oof))
sub['isFraud'] = preds

sub.to_csv("submission_xgb.csv", index=False)