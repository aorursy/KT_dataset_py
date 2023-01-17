import os

import sys

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

%matplotlib inline
working_directory_path = "/kaggle/input/ieee-fraud-detection/"

out_dir = "/kaggle/working/"

os.chdir(working_directory_path)
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

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



# Timing decorator

from functools import wraps

from time import time



def timing(f):

    @wraps(f)

    def wrapper(*args, **kwargs):

        start = time()

        result = f(*args, **kwargs)

        end = time()

        print('Elapsed time: {:.2f} Sec'.format(end-start))

        return result

    return wrapper


# Read CSV and return df

@timing

def data_load():

    train_identity = pd.read_csv("train_identity.csv")

    train_transaction = pd.read_csv("train_transaction.csv")



    test_identity = pd.read_csv("test_identity.csv")

    test_transaction = pd.read_csv("test_transaction.csv")



    train_df = pd.merge(train_identity, train_transaction, on = 'TransactionID', how='right')

    test_df = pd.merge(test_identity, test_transaction, on = 'TransactionID', how='right')

    

    print("INFO - Data load and merge complete")

    print("INFO - Train - ", train_df.shape)

    print("INFO - Submission - ", test_df.shape)

    

    return train_df, test_df



@timing

def data_info(df):

    columns = df.columns

    

# Create Metadata object for df

@timing

def data_df_metadata(df):

    

    total_rows = len(df.index)

    na_count = df.isna().sum().to_list()

    na_pct =  list(map(lambda x: round(x / total_rows * 100) , na_count))

    unique_count = list(map(lambda x: len(df[x].unique()) , df.columns))

    memory = round((df.memory_usage(index=False, deep=False)/1024**2),2).to_list()

    

    print(len(df.columns.to_list()), len(df.dtypes.to_list()), len(df.isna().sum().to_list()), len(na_pct), len(unique_count), len(memory))

    

    metadata_dict = {'column': df.columns.to_list(), 

                     'dtype': df.dtypes.to_list(),

                     'na_count': na_count, 

                     'na_pct': na_pct,

                     'unique_count': unique_count,

                     'memory': memory}

    return pd.DataFrame(metadata_dict)



# Save data as pickle

def data_save_pkl(train_df, submission_df):

    train_df.to_pickle(out_dir + "train_df.pkl")

    print("INFO - train_df saved as pkl")

    submission_df.to_pickle(out_dir + "submission_df.pkl")

    print("INFO - submission_df saved as pkl")



# Print Categorical info

def data_print_categorical(df):

    for col in df.columns:

        if (df[col].dtype == 'object'):

            print("---------- ---------- ----------")

            print(df[col].describe())

            print("----------")

            print(df[col].value_counts(dropna=False))

            

            

# Load equal no.of target class data

@timing

def data_load_equal_target_count(train_df):

    positive_train_df = train_df[train_df.isFraud == 1]

    negative_train_df = train_df[train_df.isFraud == 0].sample(n=positive_train_df.shape[0], random_state=10)

    

    combine_df = positive_train_df.append(negative_train_df).sample(frac=1, random_state=10)

    print(positive_train_df.shape, negative_train_df.shape, combine_df.shape)

    

    return combine_df
input_train_df, input_test_df = data_load()
train_df = data_load_equal_target_count(input_train_df)

test_df = input_test_df

# del input_train_df
# sample_train_df = input_train_df.sample(frac=0.2)
# # sample_train_df.columns.to_list()



# cols_show = ['TransactionDT',

#  'TransactionAmt',

#  'ProductCD',

#  'card1',

#  'card2',

#  'card3',

#  'card4',

#  'card5',

#  'card6',

#  'addr1',

#  'addr2',

#  'dist1',

#  'dist2',

#  'P_emaildomain',

#  'R_emaildomain']



# sample_train_df[cols_show]



# sample_train_df['D5'].unique()

# # sample_train_df['id_03'].hist()
# train_df = reduce_mem_usage(train_df)

# test_df = reduce_mem_usage(test_df)



# data_save_pkl(train_df, submission_df)
# msno.bar(train_df, figsize=(60, 10))

# msno.matrix(train_df, figsize=(40, 10))
# train_df.info()
# Drop columns which have > 10% NAN

metadata_df = data_df_metadata(train_df)

columns_to_drop = metadata_df[metadata_df['na_pct']>53].column.to_list()

train_df = train_df.drop(columns_to_drop, axis=1)

# columns_to_drop.remove('isFraud')

test_df = test_df.drop(columns_to_drop, axis=1)



metadata_df = data_df_metadata(train_df)

# submission_df.info()

submission_df = test_df['TransactionID']

submission_df
# data_print_categorical(train_df)

train_df = train_df.drop('TransactionID', axis=1)

test_df = test_df.drop('TransactionID', axis=1)



train_df = train_df.drop('P_emaildomain', axis=1)

test_df = test_df.drop('P_emaildomain', axis=1)
# train_df
data_print_categorical(train_df)
# train_df = reduce_mem_usage(train_df)

# test_df = reduce_mem_usage(test_df)
train_df.fillna(-999, inplace=True)

test_df.fillna(-999, inplace=True)
from sklearn.preprocessing import LabelEncoder 



metadata_df = data_df_metadata(train_df)

col_list = metadata_df[metadata_df['dtype'] == 'object'].column.to_list()

print(col_list)



encoders = {}



@timing

def setup_encoders(df, col_list):

    encoders = {}

    for col in col_list:

        print('processing: ', col)

        LE = LabelEncoder() 

        encoders[col] = LE.fit(list(df[col].astype(str).values))

    return encoders



@timing

def encode_data(df, col_list, encoders):

    for col in col_list:

        print('processing: ', col)

        LE = encoders[col]

        df[col] = LE.transform(list(df[col].astype(str).values)) 

    return df



encoders = setup_encoders(train_df, col_list)



train_df = encode_data(train_df, col_list, encoders)

test_df = encode_data(test_df, col_list, encoders)
test_df
# train_df

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(train_df, test_size = 0.4, random_state = 0)
Y_train = X_train['isFraud']

X_train = X_train.drop(['isFraud'], axis=1)



Y_test = X_test['isFraud']

X_test = X_test.drop(['isFraud'], axis=1)
# X_train
# from sklearn.preprocessing import RobustScaler



# scaler = RobustScaler()



# def data_scalar(train_df, test_df, sub_df):

#     scaled_train_df = scaler.fit_transform(train_df)

#     scaled_test_df = scaler.fit_transform(test_df)

#     scaled_sub_df = scaler.fit_transform(sub_df)

    

#     return pd.DataFrame(scaled_train_df), pd.DataFrame(scaled_test_df), pd.DataFrame(scaled_train_df), 



# X_train, X_test, test_df = data_scalar(X_train, X_test, test_df)
# %%time



# from sklearn import decomposition



# def run_pca(df):

#     pca = decomposition.PCA(n_components=20)

#     pca.fit(df)

#     print(pca.explained_variance_ratio_)

#     return pca.transform(df)

# X_pca = run_pca(X_train)

# X_sub_pca = run_pca(submission_df)
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

import lightgbm as lgb



@timing

def get_model_MNB(X, target, alpha=1):

    model = MultinomialNB(alpha=alpha).fit(X, target)

    return model



@timing

def get_model_BNB(X, target, alpha=1):

    model = BernoulliNB(alpha=alpha).fit(X, target)

    return model



@timing

def get_model_GNB(X, target, alpha=1):

    model = GaussianNB().fit(X, target)

    return model



@timing

def get_model_RF(X, target):

    model = RandomForestClassifier()

    return model.fit(X, target)



@timing

def get_model_XGB_simple(X, target):

    model = xgb.XGBClassifier()

    return model.fit(X, target)



@timing

def get_model_XGB_custom(X, target):

    model = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)

    return model.fit(X, target)



@timing

def get_model_lgb(lgb_train, lgb_val):

    parameters = {

        'application': 'binary',

        'objective': 'binary',

        'metric': 'auc',

        'is_unbalance': 'true',

        'boosting': 'gbdt',

        'num_leaves': 31,

        'feature_fraction': 0.5,

        'bagging_fraction': 0.5,

        'bagging_freq': 20,

        'learning_rate': 0.05,

        'verbose': 2

    }

    model = lgb.train(parameters,

                       lgb_train,

                       valid_sets=lgb_val,

                       num_boost_round=2000,

                       early_stopping_rounds=100)

    return model



@timing

def get_model_SGD(X, target):

    model = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)

    return model.fit(X, target)



@timing

def run_exp(model, X_test, Y_test):

    prediction = model.predict_proba(X_test)

    score = prediction[:, 1].round(1)

    plt.hist(score)

#     print(metrics.classification_report(Y_test, score))

    print("ROC_AUC: ", metrics.roc_auc_score(Y_test, score))

    return model, prediction



@timing

def run_exp_lgb(model, X_test, Y_test):

    prediction = model.predict(X_test, num_iteration=model.best_iteration)

    print(prediction)

    plt.hist(prediction)

    return model, prediction



@timing

def run_all_exp():

    model_BernoulliNB  = get_model_BNB(X_train, Y_train, 1)

    print("BernoulliNB", run_exp(model_BernoulliNB, X_test, Y_test))

    print("---------- ----------")

    model_RF  = get_model_RF(X_train, Y_train)

    print("RF", run_exp(model_RF, X_test, Y_test))

    print("---------- ----------")

    model_SGD  = get_model_SGD(X_train, Y_train)

    print("SGD", run_exp(model_SGD, X_test, Y_test))

    print("---------- ----------")

#     model_RF  = get_model_RF(X_train, Y_train)

#     print("RF", run_exp(model_RF, X_test, Y_test))

#     print("---------- ----------")

#     model_RF  = get_model_RF(X_train, Y_train)

#     print("RF", run_exp(model_RF, X_test, Y_test))

#     print("---------- ----------")



@timing

def save_submission(model, X_test, X_sub, name, isLGB = False):

    prediction = []

    if (isLGB):

        prediction = model.predict(X_test)

        score = prediction.round(1)

    else:

        prediction = model.predict_proba(X_test)

        score = prediction[:, 1].round(1)

    print(score)

    print(len(score))

    plt.hist(score)

    submission_dict = {'TransactionID': X_sub, 'isFraud': score}

    out_df = pd.DataFrame(submission_dict) 



    # saving the dataframe 

    os.chdir(out_dir)

    out_df.to_csv(name, index=False)

    return score

    
model_BernoulliNB  = get_model_BNB(X_train, Y_train, 1)

_, pred = run_exp(model_BernoulliNB, X_test, Y_test)
model_RF  = get_model_RF(X_train, Y_train)

_, pred = run_exp(model_RF, X_test, Y_test)
# get_model_XGB_simple

# model_XGB_simple  = get_model_XGB_simple(X_train, Y_train)

# _, pred = run_exp(model_XGB_simple, X_test, Y_test)
model_XGB_custom  = get_model_XGB_custom(X_train, Y_train)

_, pred = run_exp(model_XGB_custom, X_test, Y_test)
lgb_train = lgb.Dataset(X_train, Y_train)

lgb_eval = lgb.Dataset(X_test, Y_test)



model_lgb = get_model_lgb(lgb_train, lgb_eval)

_, pred = run_exp_lgb(model_lgb, X_test, Y_test)
# model_SGD  = get_model_SGD(X_train, Y_train)

# _, pred = run_exp(model_SGD, X_test, Y_test)
# run_all_exp()
score = save_submission(model_XGB_custom, test_df, submission_df, '24_XGB_2204_submsission.csv')
score = save_submission(model_lgb, test_df, submission_df, '24_LGB_2204_submsission.csv', True)
# import autosklearn.classification

# import sklearn.model_selection

# import sklearn.metrics



# automl = autosklearn.classification.AutoSklearnClassifier()

# automl.fit(X_train, Y_train)

# Y_hat = automl.predict(X_test)

# print("Accuracy score", sklearn.metrics.accuracy_score(Y_test, Y_hat))