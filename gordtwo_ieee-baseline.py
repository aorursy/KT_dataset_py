import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import datetime 

import lightgbm as lgb

from sklearn import preprocessing

import xgboost as xgb



import catboost as cb

from catboost import CatBoostClassifier, Pool





from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler 

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression



import os

import gc

print(os.listdir("../input"))
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
# for col in train.columns:

  #  if train[col].dtype=='float64': train[col] = train[col].astype('float32')

   # if train[col].dtype=='int64': train[col] = train[col].astype('int32')
%%time

train_transaction = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID'))

test_transaction = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID'))



train_identity = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID'))

test_identity = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID'))
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
train.info()
list(train.columns) 
# train['id_11'].value_counts(dropna=False, normalize=True).head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(tt)
display(missing_data(train), missing_data(test))
# potentially removing void columns



#def corret_card_id(x): 

 #   x=x.replace('.0','')

  #  x=x.replace('-999','nan')

   # return x



def define_indexes(df):

    

    # create date column

    START_DATE = '2017-12-01'

    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

    

    df['year'] = df['TransactionDT'].dt.year

    df['month'] = df['TransactionDT'].dt.month

    df['dow'] = df['TransactionDT'].dt.dayofweek

    df['hour'] = df['TransactionDT'].dt.hour

    df['day'] = df['TransactionDT'].dt.day

   

    # create card ID 

    cards_cols= ['card1', 'card2', 'card3', 'card5']

    for card in cards_cols: 

        if '1' in card: 

            df['card_id']= df[card].map(str)

        else : 

            df['card_id']+= ' '+df[card].map(str)

    

    # small correction of the Card_ID

    df['card_id']=df['card_id'].apply(corret_card_id)



define_indexes(train)

define_indexes(test)



# columns that only have ONE value

one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]



# columns that have more than 90% missing values

many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]

many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]



# columns that have values showing up more than 90% of the time

big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[1] > 0.9]

big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[1] > 0.9]



# AND... we don't want them

cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))



cols_to_drop.remove('isFraud')



train.drop(cols_to_drop, axis=1, inplace=True)

test.drop(cols_to_drop, axis=1, inplace=True)
# anything 3 sd away / boxplot

def CalcOutliers(df_num): 



    # calculating mean and std of the array

    data_mean, data_std = np.mean(df_num), np.std(df_num)



    # seting the cut line to both higher and lower values

    # You can change this value

    cut = data_std * 3



    #Calculating the higher and lower cut values

    lower, upper = data_mean - cut, data_mean + cut



    # creating an array of lower, higher and total outlier values 

    outliers_lower = [x for x in df_num if x < lower]

    outliers_higher = [x for x in df_num if x > upper]

    outliers_total = [x for x in df_num if x < lower or x > upper]



    # array without outlier values

    outliers_removed = [x for x in df_num if x > lower and x < upper]

    

    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers

    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers

    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides

    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values

    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points

    

    return
train['nulls'] = train.isnull().sum(axis=1)

test['nulls'] = test.isnull().sum(axis=1)

#from impyute.imputation.cs import mice



#imputed = mice(train.values)

#mice_ = imputed[:, :]

# in case the imputed value is smaller than 0

#for col in train.columns:

 #   if col < 0:

        

#id_38 = [0 if id_38 < 0 else id_38 for id_num in id_38]

categorical_features = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",

                        "addr1", "addr2", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", "M5",

                        "M6", "M7", "M8", "M9", "DeviceType", "DeviceInfo", "id_12", "id_13", "id_14",

                        "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22", "id_23",

                        "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32", 

                        "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"]
for col in categorical_features:

    if col in train.columns():

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col]  = le.transform(list(test[col].astype(str).values))

# Alternate code on Kaggle

#for f in train.columns:

 #   if train[f].dtype=='object' or test[f].dtype=='object': 

  #      lbl = preprocessing.LabelEncoder()

   #     lbl.fit(list(train[f].values) + list(test[f].values))

    #    train[f] = lbl.transform(list(train[f].values))

     #   test[f] = lbl.transform(list(test[f].values)) 

                                

                                  
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



# In order to save some memory for Kaggle 



del train_transaction, train_identity, test_transaction, test_identity



target = train['isFraud'].copy()



# prepare data for our model

X_train = train.drop('isFraud', axis=1)

X_train.drop('TransactionDT', axis=1, inplace=True)

X_test = test.drop('TransactionDT', axis=1)



del train, test
for f in X_train.select_dtypes(include='category').columns.tolist() + X_train.select_dtypes(include='object').columns.tolist():

    lbl = LabelEncoder()

    lbl.fit(list(X_train[f].values) + list(X_test[f].values))

    X_train[f] = lbl.transform(list(X_train[f].values))

    X_test[f] = lbl.transform(list(X_test[f].values))  
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

          'random_state': 47

         }
splits = 5

folds = KFold(n_splits = splits)

oof = np.zeros(len(X_train))

predictions = np.zeros(len(X_test))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, target.values)):

    print("Fold {}".format(fold_))

    train_df, y_train_df = X_train.iloc[trn_idx], target.iloc[trn_idx]

    valid_df, y_valid_df = X_train.iloc[val_idx], target.iloc[val_idx]

    

    trn_data = lgb.Dataset(train_df, label=y_train_df)

    val_data = lgb.Dataset(valid_df, label=y_valid_df)

    

    clf = lgb.train(params,

                    trn_data,

                    10000,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=500,

                    early_stopping_rounds=500)



    pred = clf.predict(valid_df)

    oof[val_idx] = pred

    print( "  auc = ", roc_auc_score(y_valid_df, pred) )

    predictions += clf.predict(X_test) / splits
# submission

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

sample_submission = sample_submission.reset_index()

sample_submission["isFraud"] = predictions

sample_submission.to_csv("lgb_sub_baseline.csv", index=False)