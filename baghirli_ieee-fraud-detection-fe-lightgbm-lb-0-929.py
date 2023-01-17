import os

import re

import sys

import numpy as np

import pandas as pd

import random

import gc



from datetime import datetime, timedelta

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.pipeline import FeatureUnion, Pipeline 

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder



import lightgbm as lgb

from sklearn import metrics

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import StratifiedKFold



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



import warnings

warnings.filterwarnings("ignore")



pd.options.display.max_columns = 1000

pd.options.display.max_rows = 1000

pd.options.display.max_colwidth = -1



def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



RANDOM_STATE = 42

seed_everything(seed=RANDOM_STATE)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: 

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_transaction_full = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

test_transaction_full = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")



len_train = train_transaction_full.shape[0]

len_test = test_transaction_full.shape[0]



train_index = pd.RangeIndex(start=0, stop=len_train, step=1)

test_index = pd.RangeIndex(start=len_train, stop=len_train + len_test, step=1)



train_transaction_full.index = train_index

test_transaction_full.index = test_index



train_transaction_full = reduce_mem_usage(train_transaction_full)

test_transaction_full = reduce_mem_usage(test_transaction_full)
def DT(x):

    return start_date + timedelta(seconds=x)



def vestas_nan_count(row):

    return row.isnull().sum()



def M_nan_count(row):

    return row.isnull().sum()



def ADDR_nan_count(row):

    return row.isnull().sum()



# def merge_columns(cols, out_col, tr, tt):

#     len_tr = len(tr)

#     len_tt = len(tt)



#     temp_tr = pd.Series(["" for i in range(len_tr)])

#     temp_tt = pd.Series(["" for i in range(len_tt)])  



#     for col in cols:

#         temp_tr = temp_tr + tr[col].astype(str)

#         temp_tt = temp_tt + tt[col].astype(str)

    

#     tr[out_col] = temp_tr

#     tt[out_col] = temp_tt



def merge_columns(col1, col2, out_col, tr, tt):

    tr[out_col] = tr[col1] + tr[col2]

    tt[out_col] = tt[col1] + tt[col2]



def ona_to_many(col1, col2, out_col, tr, tt):

    joint = tr[[col1, col2]].append(tt[[col1, col2]], ignore_index=True)

    joint[out_col] = 0



    temp = joint[joint[col1].notnull()]

    joint.loc[temp.index, out_col] = temp.groupby([col1])[col2].transform(lambda x: x.nunique())



    tr[out_col] = joint.loc[tr.index, out_col]

    tt[out_col] = joint.loc[tt.index, out_col]



    del joint, temp

    gc.collect()

    

def add_D_lags(df, columns):

    for col in columns:

        temp = df[col] - df.groupby(['WeekOfYear'])[col].transform(func=np.mean)

        min_col = temp.min()

        df[col + "_WeeklyAveDiff"] = temp.apply(lambda x: x + np.abs(min_col)).apply(np.log1p)

        print("Processed: ", col)
class VestasImputer(BaseEstimator, TransformerMixin):

    def __init__(self, vestas_columns):

        self._vestas_features = vestas_columns



    def fit(self, X, y = None):

        return self

    

    def transform(self, X, y = None):

        gc.collect()

        if X[self._vestas_features].isnull().sum().sum() > 0:

            fill_value = np.around(X[self._vestas_features].min() - 2)

            X[self._vestas_features] = X[self._vestas_features].fillna(fill_value)

        return X



class VestasScaler(BaseEstimator, TransformerMixin):

    def __init__(self, vestas_columns):

        self._vestas_features = vestas_columns

        self._scaler = MinMaxScaler()



    def fit(self, X, y = None):

        gc.collect()

        self._scaler.fit(X[self._vestas_features])

        return self

    

    def transform(self, X, y = None):

        gc.collect()

        X[self._vestas_features] = self._scaler.transform(X[self._vestas_features])

        return X



class VestasPCATransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vestas_columns, n_components=20, random_state=RANDOM_STATE):

        self._vestas_features = vestas_columns

        self._pca = PCA(n_components=n_components, random_state=random_state)

        

    def fit(self, X, y = None):

        gc.collect()

        self._pca.fit(X[self._vestas_features])

        return self

    

    def transform(self, X, y = None):

        gc.collect()

        principal_df = pd.DataFrame(self._pca.transform(X[self._vestas_features]))

        principal_df.index = X.index

        principal_df.rename(columns=lambda x: "PCA_V" + str(x), inplace=True)

        X.drop(self._vestas_features, axis=1, inplace=True)

        return pd.concat([X, principal_df], axis=1)
# DT related derived features



START_DATE = "2017-12-31"

start_date = datetime.strptime(START_DATE, "%Y-%m-%d")



train_transaction_full['DT'] = train_transaction_full['TransactionDT'].apply(lambda x: DT(x))

test_transaction_full['DT'] = test_transaction_full['TransactionDT'].apply(lambda x: DT(x))



train_transaction_full['DT_M'] = train_transaction_full['DT'].dt.month.astype(np.int8)

test_transaction_full['DT_M'] = test_transaction_full['DT'].dt.month.astype(np.int8)



train_transaction_full['WeekOfYear'] = train_transaction_full['DT'].dt.week.astype(np.int8)

test_transaction_full['WeekOfYear'] = test_transaction_full['DT'].dt.week.astype(np.int8)



train_transaction_full['WeekOfMonth'] = train_transaction_full['DT'].dt.week.apply(lambda week: (week - 1) % 4)

test_transaction_full['WeekOfMonth'] = test_transaction_full['DT'].dt.week.apply(lambda week: (week - 1) % 4)



train_transaction_full['DayOfWeek'] = train_transaction_full['DT'].dt.dayofweek

test_transaction_full['DayOfWeek'] = test_transaction_full['DT'].dt.dayofweek



train_transaction_full['HourOfDay'] = train_transaction_full['DT'].dt.hour

test_transaction_full['HourOfDay'] = test_transaction_full['DT'].dt.hour



train_transaction_full['HourOfWeek'] = (train_transaction_full['DayOfWeek'] - 1) * 24 + train_transaction_full['HourOfDay']

test_transaction_full['HourOfWeek'] = (test_transaction_full['DayOfWeek'] - 1) * 24 + test_transaction_full['HourOfDay']
vestas_columns = list()

for i, col in enumerate(list(train_transaction_full.columns)):

    if col.startswith("V"):

        vestas_columns.append(col)



# train_transaction_full["vestas_nan_count"] = train_transaction_full[vestas_columns].apply(func=vestas_nan_count, axis=1, result_type='reduce')

# test_transaction_full["vestas_nan_count"] = test_transaction_full[vestas_columns].apply(func=vestas_nan_count, axis=1, result_type='reduce')



bins = [0,1,2,3,4,5,6,7,8,9,10,np.inf]

labels = [1,2,3,4,5,6,7,8,9,10,11]



train_transaction_full['vestas_mean'] = train_transaction_full[vestas_columns].mean(axis=1).apply(np.log1p)

train_transaction_full['vestas_std'] = train_transaction_full[vestas_columns].std(axis=1).apply(np.log1p)

train_transaction_full['vestas_mean_label'] = pd.cut(train_transaction_full['vestas_mean'], bins=bins, labels=labels).astype(np.int8)

train_transaction_full['vestas_std_label'] = pd.cut(train_transaction_full['vestas_std'], bins=bins, labels=labels).astype(np.int8)



test_transaction_full['vestas_mean'] = test_transaction_full[vestas_columns].mean(axis=1).apply(np.log1p)

test_transaction_full['vestas_std'] = test_transaction_full[vestas_columns].std(axis=1).apply(np.log1p)

test_transaction_full['vestas_mean_label'] = pd.cut(test_transaction_full['vestas_mean'], bins=bins, labels=labels).astype(np.int8)

test_transaction_full['vestas_std_label'] = pd.cut(test_transaction_full['vestas_std'], bins=bins, labels=labels).astype(np.int8)



train_transaction_full['VV313'] = train_transaction_full["V313"]

train_transaction_full['VV307'] = train_transaction_full["V307"]

train_transaction_full['VV310'] = train_transaction_full["V310"]



test_transaction_full['VV313'] = test_transaction_full["V313"]

test_transaction_full['VV307'] = test_transaction_full["V307"]

test_transaction_full['VV310'] = test_transaction_full["V310"]



vestas_pipeline = Pipeline(steps = [('vestas_imputer', VestasImputer(vestas_columns)),

                                    ('vestas_scaler', VestasScaler(vestas_columns)),

                                    ('vestas_pca', VestasPCATransformer(vestas_columns))

                                   ])



gc.collect()

vestas_pipeline.fit(train_transaction_full)

train_transaction_full = vestas_pipeline.transform(train_transaction_full)

test_transaction_full = vestas_pipeline.transform(test_transaction_full)

gc.collect()
train_transaction_full.to_pickle("train_transaction.pkl")

test_transaction_full.to_pickle("test_transaction.pkl")
train_transaction_full = pd.read_pickle("train_transaction.pkl")

test_transaction_full = pd.read_pickle("test_transaction.pkl")
M_columns = list()

for i, col in enumerate(list(train_transaction_full.columns)):

    if col.startswith("M"):

        M_columns.append(col)



# train_transaction_full["M_nan_count"] = train_transaction_full[M_columns].apply(func=M_nan_count, axis=1, result_type='reduce')

# test_transaction_full["M_nan_count"] = test_transaction_full[M_columns].apply(func=M_nan_count, axis=1, result_type='reduce')
# merge_columns(['card1', 'card2'], 'uid1', train_transaction_full, test_transaction_full)

# merge_columns(['uid1', 'card3', 'card5'], 'uid2', train_transaction_full, test_transaction_full)

# merge_columns(['uid2', 'addr1', 'addr2'], 'uid3', train_transaction_full, test_transaction_full)

# merge_columns(['card4', 'card6'], 'card4_card6', train_transaction_full, test_transaction_full)

# merge_columns(['addr1', 'addr2'], 'addr1_addr2', train_transaction_full, test_transaction_full)

# merge_columns(['P_emaildomain', 'R_emaildomain'], 'P_emaildomain_R_emaildomain', train_transaction_full, test_transaction_full)

# merge_columns(['ProductCD', 'card6'], 'Prod_card6', train_transaction_full, test_transaction_full)



merge_columns('card4', 'card6', 'card4_card6', train_transaction_full, test_transaction_full)

merge_columns('addr1', 'addr2', 'addr1_addr2', train_transaction_full, test_transaction_full)

merge_columns('P_emaildomain', 'R_emaildomain', 'P_emaildomain_R_emaildomain', train_transaction_full, test_transaction_full)

merge_columns('ProductCD', 'card6', 'Prod_card6', train_transaction_full, test_transaction_full)
# train_transaction_full["ADDR_nan_count"] = train_transaction_full[["addr1", "addr2"]].apply(func=ADDR_nan_count, axis=1, result_type='reduce')

# test_transaction_full["ADDR_nan_count"] = test_transaction_full[["addr1", "addr2"]].apply(func=ADDR_nan_count, axis=1, result_type='reduce')
ona_to_many("addr1", "addr2", "addr1_onemany_addr2", train_transaction_full, test_transaction_full)

ona_to_many("P_emaildomain", "R_emaildomain", "P_onemany_R", train_transaction_full, test_transaction_full)

ona_to_many("card1", "P_emaildomain", "card1_onemany_P", train_transaction_full, test_transaction_full)

ona_to_many("card1", "addr1", "card1_onemany_addr1", train_transaction_full, test_transaction_full)
bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,np.inf]

labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]



train_transaction_full['dist1_label'] = pd.cut(train_transaction_full['dist1'].apply(np.log1p), bins=bins, labels=labels).astype(np.int8)

train_transaction_full['dist2_label'] = pd.cut(train_transaction_full['dist2'].apply(np.log1p), bins=bins, labels=labels).astype(np.int8)



test_transaction_full['dist1_label'] = pd.cut(test_transaction_full['dist1'].apply(np.log1p), bins=bins, labels=labels).astype(np.int8)

test_transaction_full['dist2_label'] = pd.cut(test_transaction_full['dist2'].apply(np.log1p), bins=bins, labels=labels).astype(np.int8)



dist_columns = ['dist1', 'dist2']

dist_aggr_cols = ['card1', 'P_emaildomain', 'addr1', 'addr2']



for feat_col in dist_columns:

    for aggr_col in dist_aggr_cols:

        train_transaction_full[feat_col + "_by_" + aggr_col] = train_transaction_full[feat_col] - train_transaction_full.groupby([aggr_col])[feat_col].transform(func=np.mean)

        test_transaction_full[feat_col + "_by_" + aggr_col] = test_transaction_full[feat_col] - test_transaction_full.groupby([aggr_col])[feat_col].transform(func=np.mean)
bins =   [0,1,2,3,4,5,6,7,8, 9,10,11,12,13,np.inf]

labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]



train_transaction_full['TransactionAmt_label'] = pd.cut(train_transaction_full['TransactionAmt'].apply(np.log1p), bins=bins, labels=labels).astype(np.int8)

test_transaction_full['TransactionAmt_label'] = pd.cut(test_transaction_full['TransactionAmt'].apply(np.log1p), bins=bins, labels=labels).astype(np.int8)



train_transaction_full['TransactionAmt_log'] = train_transaction_full['TransactionAmt'].apply(np.log1p)

test_transaction_full['TransactionAmt_log'] = test_transaction_full['TransactionAmt'].apply(np.log1p)



TransactionAmt_columns = ['TransactionAmt']

TransactionAmt_aggr_cols = ['card1','card2','card3', 'card5', 'P_emaildomain', 'addr1', 'addr2']



for feat_col in TransactionAmt_columns:

    for aggr_col in TransactionAmt_aggr_cols:

        train_transaction_full[feat_col + "_by_" + aggr_col] = train_transaction_full[feat_col] / train_transaction_full.groupby([aggr_col])[feat_col].transform(func=np.std)

        test_transaction_full[feat_col + "_by_" + aggr_col] = test_transaction_full[feat_col] / test_transaction_full.groupby([aggr_col])[feat_col].transform(func=np.std)

        # fix the np.inf stemming from std=0 or nan

        train_transaction_full[feat_col + "_by_" + aggr_col].replace([np.inf, -np.inf], np.nan, inplace=True) 

        test_transaction_full[feat_col + "_by_" + aggr_col].replace([np.inf, -np.inf], np.nan, inplace=True) 

        print("Processed: {0} by {1}".format(feat_col, aggr_col))



train_transaction_full['TransactionAmt_cents_len'] = train_transaction_full['TransactionAmt'].astype(str).str.split(".").apply(lambda x: len(x[1]) if len(x) > 1 else 0)

test_transaction_full['TransactionAmt_cents_len'] = test_transaction_full['TransactionAmt'].astype(str).str.split(".").apply(lambda x: len(x[1]) if len(x) > 1 else 0)
freq_encoded_columns = ["card1", "card2", "card3", "card5", 'card4', 'card6', "addr1", "addr2", "addr1_addr2", "card4_card6", 

                        "Prod_card6", "P_emaildomain", "R_emaildomain", "P_emaildomain_R_emaildomain"]



target_encoded_columns = freq_encoded_columns



print("Starting Freq Encoding ...")

for col in freq_encoded_columns:

    print("freq encoding: ", col)

    temp = train_transaction_full[col].append( test_transaction_full[col], ignore_index=True)

    col_map = temp.value_counts(dropna=True, normalize=False).to_dict()

    temp_map = temp.map(col_map)

    train_transaction_full[col + "_freq"] = temp_map.loc[train_transaction_full.index]

    test_transaction_full[col + "_freq"] = temp_map.loc[test_transaction_full.index]

print("Freq Encoding Done")



# print('\n')



# print("Starting Target Encoding ...")

# for col in target_encoded_columns:

#     print("target encoding: ", col)

#     col_target_map = train_transaction_full.groupby([col])['isFraud'].mean().to_dict()

#     train_transaction_full[col + "_tenc"] = train_transaction_full[col].map(col_target_map)

#     test_transaction_full[col + "_tenc"] = test_transaction_full[col].map(col_target_map)

# print("Target Encoding Done")
C_columns = list()

for i, col in enumerate(list(train_transaction_full.columns)):

    if col.startswith("C"):

        C_columns.append(col)



train_transaction_full['C_sum'] = train_transaction_full[C_columns].apply(np.log1p).sum(axis=1)

train_transaction_full['C_mean'] = train_transaction_full[C_columns].apply(np.log1p).mean(axis=1)

train_transaction_full['C_std'] = train_transaction_full[C_columns].apply(np.log1p).std(axis=1)



test_transaction_full['C_sum'] = test_transaction_full[C_columns].apply(np.log1p).sum(axis=1)

test_transaction_full['C_mean'] = test_transaction_full[C_columns].apply(np.log1p).mean(axis=1)

test_transaction_full['C_std'] = test_transaction_full[C_columns].apply(np.log1p).std(axis=1)
D_columns = list()

for i, col in enumerate(list(train_transaction_full.columns)):

    if col.startswith("D") and col.isalpha() is False and len(col) < 4:

        D_columns.append(col)



add_D_lags(train_transaction_full, D_columns)

add_D_lags(test_transaction_full, D_columns)
numeric_columns = list(train_transaction_full.select_dtypes(include=np.number))

object_columns = list(train_transaction_full.select_dtypes(include="object"))



for num_col in numeric_columns:

    if num_col == "isFraud":

        continue

    fill_value = np.around(train_transaction_full[num_col].append(test_transaction_full[num_col], ignore_index=True).min() - 2)

    train_transaction_full[num_col] = train_transaction_full[num_col].fillna(fill_value)

    test_transaction_full[num_col] = test_transaction_full[num_col].fillna(fill_value)



for obj_col in object_columns:

    train_transaction_full[obj_col] = train_transaction_full[obj_col].fillna("unknown")

    test_transaction_full[obj_col] = test_transaction_full[obj_col].fillna("unknown")
le = LabelEncoder()

label_encoded_columns = freq_encoded_columns + M_columns + ['ProductCD']

for label_col in label_encoded_columns:

    le.fit(train_transaction_full[label_col].append(test_transaction_full[label_col], ignore_index=True).values)

    train_transaction_full[label_col] = le.transform(train_transaction_full[label_col])

    test_transaction_full[label_col] = le.transform(test_transaction_full[label_col])
gc.collect()



train_identity_full = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

test_identity_full = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")



len_train_identity = len(train_identity_full)

len_test_identity = len(test_identity_full)

test_identity_full.index = pd.RangeIndex(start=len_train_identity, stop=len_train_identity + len_test_identity, step=1)



train_identity_full = reduce_mem_usage(train_identity_full)

test_identity_full = reduce_mem_usage(test_identity_full)



train_identity_full.to_pickle("train_identity.pkl")

test_identity_full.to_pickle("test_identity.pkl")
train_identity_full = pd.read_pickle("train_identity.pkl")

test_identity_full = pd.read_pickle("test_identity.pkl")
# patch_train = train_transaction_full[["TransactionID", "isFraud"]].copy()

# train_identity_full = pd.merge(patch_train, train_identity_full, on="TransactionID", how="inner")
def get_device_id(x):

    try:

        tokens = re.split(r'\s|\/|\-|\_', x)

        return tokens[0]

    except Exception:

        return np.nan



def get_os_id(x):

    try:

        tokens = re.split(r'\s|\/', x.lower())

        return tokens[0]

    except Exception:

        return np.nan



def get_browser_id(x):

    try:

        tokens = re.split(r'\s|\/', x)

        return tokens[0]

    except Exception:

        return np.nan
train_identity_full['id_30'] = train_identity_full['id_30'].apply(lambda x: get_os_id(x))

test_identity_full['id_30'] = test_identity_full['id_30'].apply(lambda x: get_os_id(x))



train_identity_full['id_31'] = train_identity_full['id_31'].replace("Firefox/Mozilla", "mozilla")

test_identity_full['id_31'] = test_identity_full['id_31'].replace("Firefox/Mozilla", "mozilla")



train_identity_full['id_31'] = train_identity_full['id_31'].str.lower().replace("mobile", "", regex=True).str.strip()

test_identity_full['id_31'] = test_identity_full['id_31'].str.lower().replace("mobile", "", regex=True).str.strip()



train_identity_full['id_31'] = train_identity_full['id_31'].replace("", "mobile")

test_identity_full['id_31'] = test_identity_full['id_31'].replace("", "mobile")



train_identity_full['id_31'] = train_identity_full['id_31'].replace("firefox", "mozilla")

test_identity_full['id_31'] = test_identity_full['id_31'].replace("firefox", "mozilla")



train_identity_full['id_31'] = train_identity_full['id_31'].apply(lambda x: get_browser_id(x))

test_identity_full['id_31'] = test_identity_full['id_31'].apply(lambda x: get_browser_id(x))



train_identity_full['DeviceInfo'] = train_identity_full['DeviceInfo'].str.lower()

test_identity_full['DeviceInfo'] = test_identity_full['DeviceInfo'].str.lower()



train_identity_full['DeviceInfo'] = train_identity_full['DeviceInfo'].apply(lambda x: get_device_id(x))

test_identity_full['DeviceInfo'] = test_identity_full['DeviceInfo'].apply(lambda x: get_device_id(x))



train_identity_full['id_23'] = train_identity_full['id_23'].replace("IP_PROXY:", "", regex=True).str.lower()

test_identity_full['id_23'] = test_identity_full['id_23'].replace("IP_PROXY:", "", regex=True).str.lower()
categorical_ide_columns = ["id_" + str(i) for i in range(12, 39)] + ['DeviceInfo', 'DeviceType']

label_encoded_ide_columns = list(set(categorical_ide_columns) - set([]))

freq_encoded_ide_columns = ['DeviceInfo', 'id_33', 'id_31', 'id_30', 'DeviceType', 'id_23']



print("Starting Freq Encoding ...")

for col in freq_encoded_ide_columns:

    print("freq encoding: ", col)

    temp = train_identity_full[col].append(test_identity_full[col], ignore_index=True)

    col_map = temp.value_counts(dropna=True, normalize=False).to_dict()

    temp_map = temp.map(col_map)

    train_identity_full[col + "_freq"] = temp_map.loc[train_identity_full.index]

    test_identity_full[col + "_freq"] = temp_map.loc[test_identity_full.index]

print("Freq Encoding Done")



numeric_ide_columns = list(train_identity_full.select_dtypes(include=np.number))

object_ide_columns = list(train_identity_full.select_dtypes(include="object"))



for num_col in numeric_ide_columns:

    if num_col == "isFraud":

        continue

    fill_value = np.around(train_identity_full[num_col].append(test_identity_full[num_col], ignore_index=True).min() - 2)

    train_identity_full[num_col] = train_identity_full[num_col].fillna(fill_value)

    test_identity_full[num_col] = test_identity_full[num_col].fillna(fill_value)



for obj_col in object_ide_columns:

    train_identity_full[obj_col] = train_identity_full[obj_col].fillna("unknown")

    test_identity_full[obj_col] = test_identity_full[obj_col].fillna("unknown")



le = LabelEncoder()

for label_col in label_encoded_ide_columns:

    le.fit(train_identity_full[label_col].append(test_identity_full[label_col], ignore_index=True).values)

    train_identity_full[label_col] = le.transform(train_identity_full[label_col])

    test_identity_full[label_col] = le.transform(test_identity_full[label_col])
# os.remove("train_transaction.pkl"); os.remove("test_transaction.pkl")

# os.remove("train_identity.pkl"); os.remove("test_identity.pkl")

gc.collect()



train_transaction_full.to_pickle("train_transaction_clean.pkl"); test_transaction_full.to_pickle("test_transaction_clean.pkl")

train_identity_full.to_pickle("train_identity_clean.pkl"); test_identity_full.to_pickle("test_identity_clean.pkl")
train_transaction_full = pd.read_pickle("train_transaction_clean.pkl"); test_transaction_full = pd.read_pickle("test_transaction_clean.pkl")

train_identity_full = pd.read_pickle("train_identity_clean.pkl"); test_identity_full = pd.read_pickle("test_identity_clean.pkl")
try:

    train_identity_full.drop(['isFraud'], axis=1, inplace=True)

except Exception:

    pass



train = train_transaction_full.merge(train_identity_full, how="left", on="TransactionID")

train.index = train_index

del train_transaction_full, train_identity_full



test = test_transaction_full.merge(test_identity_full, how="left", on="TransactionID")

test.index = test_index

del test_transaction_full, test_identity_full



drop_columns = ["DT", "WeekOfYear", "TransactionDT", "TransactionID"] 



train.drop(drop_columns, axis=1, inplace=True)

test.drop(drop_columns, axis=1, inplace=True)



train = pd.concat([train.loc[train['isFraud']==1], train.loc[train['isFraud']==0].sample(frac=0.2, random_state=RANDOM_STATE)]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)



y_train = train.isFraud

train.drop(['isFraud'], axis=1, inplace=True)



gc.collect()
numeric_columns_merge = list(set(list(train.select_dtypes(include=np.number)) + list(test.select_dtypes(include=np.number))))

object_columns_merge = list(set(list(train.select_dtypes(include="object")) + list(test.select_dtypes(include="object"))))

print("Object columns: ", object_columns_merge)



for num_col in numeric_columns_merge:

    temp = train[num_col].append(test[num_col], ignore_index=True)

    if temp.isnull().sum() > 0: 

        fill_value = np.around(temp.min() - 4)

        train[num_col] = train[num_col].fillna(fill_value)

        test[num_col] = test[num_col].fillna(fill_value)



del temp; gc.collect()



for obj_col in object_columns_merge:

    train[obj_col] = train[obj_col].fillna("misside")

    test[obj_col] = test[obj_col].fillna("misside")
train = pd.get_dummies(data=train, columns=object_columns_merge)

test = pd.get_dummies(data=test, columns=object_columns_merge)
train.head()
def integrity_check(train, test):

    drop_cols = list()

    all_cols = set(list(train) + list(test))

    train_cols = set(list(train))

    test_cols = set(list(test))



    drop_cols = drop_cols + list(train_cols.union(test_cols) - train_cols.intersection(test_cols))



    remaining_cols = list(all_cols - set(drop_cols))



    for col in remaining_cols:

        cond1 = train[col].nunique() in [0, 1]

        if cond1 and col != "DT_M":

            drop_cols.append(col)



    print("Columns To Be Dropped: ", drop_cols)



    for d_col in drop_cols:

        try:

            train.drop([d_col], axis=1, inplace=True)

        except Exception:

            pass

        try:

            test.drop([d_col], axis=1, inplace=True)

        except Exception:

            pass



integrity_check(train, test)
ordinal_category = ['vestas_mean_label', 'vestas_std_label', 'dist1_label', 'dist2_label', 'TransactionAmt_label']

nominal_category = list(set(label_encoded_columns + label_encoded_ide_columns))



for ord_col in ordinal_category:

    train[ord_col] = pd.Categorical(train[ord_col], ordered=True)

    test[ord_col] = pd.Categorical(test[ord_col], ordered=True)



for nom_col in nominal_category:

    train[nom_col] = pd.Categorical(train[nom_col], ordered=False)

    test[nom_col] = pd.Categorical(test[nom_col], ordered=False)
[train.isnull().sum().sum(), test.isnull().sum().sum()]
assert set(list(train)) == set(list(test))
NFOLDS = 5



lgb_params = {

                    'objective': 'binary',

                    'boosting_type': 'gbdt',

                    'metric': 'auc',

                    'n_jobs': -1,

                    'learning_rate': 0.007,

                    'num_leaves': 2**8,

                    'max_depth': -1,

                    'tree_learner': 'serial',

                    'colsample_bytree': 0.7,

                    'subsample_freq': 1,

                    'subsample': 0.8,

                    'n_estimators': 1, # 3000

                    'max_bin': 255,

                    'verbose': -1,

                    'seed': RANDOM_STATE,

                    'early_stopping_rounds': 100,

                    'is_unbalance': True

                }



# folds = GroupKFold(n_splits=NFOLDS)

folds = StratifiedKFold(n_splits=NFOLDS)



split_groups = train['DT_M']

feature_cols = list(set(list(train) + list(test)) - set(['DT_M']))

X_train = train[feature_cols]

y_train = y_train

X_test = test[feature_cols]

assert y_train.shape[0] == X_train.shape[0]



predictions = np.zeros(len(X_test))

oof = np.zeros(len(X_train))

score = 0



# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train, groups=split_groups)):

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):

    print('Fold:',fold_)

    tr_x, tr_y = X_train.iloc[trn_idx,:], y_train[trn_idx]

    vl_x, vl_y = X_train.iloc[val_idx,:], y_train[val_idx]



    print(len(tr_x), len(vl_x))

    tr_data = lgb.Dataset(tr_x, label=tr_y)

    vl_data = lgb.Dataset(vl_x, label=vl_y)  



    estimator = lgb.train(

        lgb_params,

        tr_data,

        valid_sets = [tr_data, vl_data],

        verbose_eval = 200

    )   



    pp_p = estimator.predict(X_test)

    predictions += pp_p/NFOLDS



    vv_v = estimator.predict(vl_x)

    score += metrics.roc_auc_score(vl_y, vv_v)/NFOLDS

    

    oof_preds = estimator.predict(vl_x)

    oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())



    feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X_train.columns), reverse=True), columns=['Value','Feature'])

    print(feature_imp)



    del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data

    gc.collect()
print('OOF AUC:', metrics.roc_auc_score(y_train, oof))

print('Mean CV AUC:', score)
print(classification_report(y_train, (pd.Series(oof) > 0.50).astype(np.int8)))

tn, fp, fn, tp = confusion_matrix(y_train, (pd.Series(oof) > 0.50).astype(np.int8)).ravel()

print(tn, fp, fn, tp)
submission = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")

submission['isFraud'] = predictions

submission.to_csv("submission.csv", index=False)
!pip install kaggle
!rm -rf /tmp/.kaggle
!mkdir /tmp/.kaggle/
!touch /tmp/.kaggle/kaggle.json
with open("/tmp/.kaggle/kaggle.json", "w") as fd:

    fd.write("""{"username":"baghirli","key":"hash"}""")
!chmod 600 /tmp/.kaggle/kaggle.json
!cat /tmp/.kaggle/kaggle.json
# !kaggle competitions submit -c ieee-fraud-detection -f submission.csv -m "lightgbm"