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

tmp_dir = "/kaggle/input/tmp/"

os.chdir(working_directory_path)
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



# Print Categorical info

def data_print_categorical(df):

    for col in df.columns:

        if (df[col].dtype == 'object'):

            print("---------- ---------- ----------")

            print(df[col].describe())

            print("----------")

            print(df[col].value_counts(dropna=False))
input_train_df, input_test_df = data_load()
cols_low_imp = ['V111', 'V108', 'V32', 'V16', 'V22', 'V121', 'V116', 'V106', 'V18', 'V299', 'V58', 'V298', 'V104', 'V115', 'V31', 'V72', 'V71', 'V109', 'V14', 'V79']



input_train_df = input_train_df.drop(cols_low_imp, axis=1)

input_test_df = input_test_df.drop(cols_low_imp, axis=1)
# Drop columns which have > 50% NAN

metadata_df = data_df_metadata(input_train_df)

columns_to_drop = metadata_df[metadata_df['na_pct']>50].column.to_list()

input_train_df = input_train_df.drop(columns_to_drop, axis=1)

input_test_df = input_test_df.drop(columns_to_drop, axis=1)



metadata_df = data_df_metadata(input_train_df)
submission_df = input_test_df['TransactionID']

submission_df.shape
# Drop unwanted columns TransactionID, P_emaildomain

input_train_df = input_train_df.drop('TransactionID', axis=1)

input_test_df = input_test_df.drop('TransactionID', axis=1)



input_train_df = input_train_df.drop('P_emaildomain', axis=1)

input_test_df = input_test_df.drop('P_emaildomain', axis=1)
data_print_categorical(input_train_df)
# from matplotlib import pyplot as plt

# import seaborn as sns

# plt.figure(figsize=(16,6))



# # target_0 = input_train_df[input_train_df['isFraud'] == 0].sample(frac=0.3)

# # target_1 = input_train_df[input_train_df['isFraud'] == 1]

# col = 'C4'



# # ax = sns.distplot(target_0[col], hist=False, rug=False)

# # ax = sns.distplot(target_1[col], hist=False, rug=False)

# # df = input_train_df.sample(frac=0.5)

# ax = sns.violinplot(x="isFraud", y=col, data=df)





# print(target_0[col].count(), target_0[col].mean())

# print(target_1[col].count(), target_1[col].mean())
input_train_df.fillna(-999, inplace=True)

input_test_df.fillna(-999, inplace=True)
# Label Encode categorical columns



from sklearn.preprocessing import LabelEncoder 



metadata_df = data_df_metadata(input_train_df)

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



encoders = setup_encoders(input_train_df, col_list)



input_train_df = encode_data(input_train_df, col_list, encoders)

input_test_df = encode_data(input_test_df, col_list, encoders)
print(input_train_df.shape)

print(input_test_df.shape)
# Split input dataframe into n sets and create n models



NUM_MODELS = 10



@timing

def data_load_multi_split(train_df):

    positive_train_df = train_df[train_df.isFraud == 1]

    negative_train_df = train_df[train_df.isFraud == 0].sample(frac=1, random_state=10)

    

    negative_train_df_array = np.array_split(negative_train_df,  NUM_MODELS)

    

    result_df_array = []

    

    for df in negative_train_df_array:

        negative_train_df_sample = df.sample(n=22000, random_state=10)

        combine_df = negative_train_df_sample.append(positive_train_df).sample(frac=1, random_state=10)

        result_df_array.append(combine_df)

    

    return result_df_array
# isFraud==1 samples = 20k, isFraud==1 samples = 560k

# Create n sets of 40k records with same isFraud==1 and different isFraud==0 records



input_df_array = data_load_multi_split(input_train_df)
print(len(input_df_array))

print(input_df_array[0].shape)
from sklearn.model_selection import train_test_split



def get_train_test_split(df_array):

    result_array = []

    for df in df_array:

        X_train, X_test = train_test_split(df, test_size = 0.4, random_state = 0)

        Y_train, X_train, Y_test, X_test = X_train['isFraud'], X_train.drop(['isFraud'], axis=1), X_test['isFraud'], X_test.drop(['isFraud'], axis=1)

        result_array.append((Y_train, X_train, Y_test, X_test))

        

    return result_array
train_test_split_array = get_train_test_split(input_df_array)
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

    model = xgb.XGBClassifier(n_estimators=250, max_depth=8, learning_rate=0.2, subsample=0.8, nthread=4)

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

    print("ROC_AUC: ", metrics.roc_auc_score(Y_test, score))

    return model, prediction



@timing

def run_exp_lgb(model, X_test, Y_test):

    prediction = model.predict(X_test, num_iteration=model.best_iteration)

    print(prediction)

    plt.hist(prediction)

    return model, prediction



@timing

def run_exp_multi_set(train_test_split_array):

    set_index = 0

    models = []

    

    for (Y_train, X_train, Y_test, X_test) in train_test_split_array:

        print("processing model for index: ", set_index)

        

        model = get_model_XGB_custom(X_train, Y_train)

        _, pred = run_exp(model, X_test, Y_test)

        

        models.append(model)

        set_index = set_index + 1

        

    return models



def save_submission_multi_set(models, name):

    

    output_file_names = []

    

    for index, model in enumerate(models):

        X_test = train_test_split_array[index][3]

        prediction = model.predict_proba(input_test_df)

        score = prediction[:, 1].round(3)

        plt.hist(score)

        plt.show()

        submission_dict = {'TransactionID': submission_df, 'isFraud': score}

        out_df = pd.DataFrame(submission_dict)



        # saving the dataframe 

        os.chdir(out_dir)

        file_name = name+str(index)+'.csv'

        out_df.to_csv(file_name, index=False)

        output_file_names.append(file_name)

        print('----- processed: ', file_name)

        

    return output_file_names

    

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

    
# We are only using XGB model for this run. check run_exp_multi_set() method above.

models = run_exp_multi_set(train_test_split_array)
from xgboost import plot_importance, plot_tree

import matplotlib.pyplot as plt



# plot feature importance

fig, ax = plt.subplots(figsize=(14, 40))

plot_importance(models[0], ax=ax)

plt.show()
# create n various of submission csv. we will combine these outputs later.

# plot hist to visualize variations in each model.

output_file_names = save_submission_multi_set(models, 'multi_set_5OCT_')
# read all individual submission csv files and calculate mean for final submission.



def combine_predictions(output_file_names):

    sub_dfs = []

    for file in output_file_names:

        df = pd.read_csv(file)

        df = df.set_index('TransactionID')

        sub_dfs.append(df)

        

    combine_df = pd.concat(sub_dfs)

    combine_df = combine_df.groupby(level=0).mean()

    combine_df['TransactionID'] = combine_df.index

    return combine_df



def save_combine_pred_csv(df, name):

    # saving the dataframe 

    df.to_csv(name, columns=["TransactionID", "isFraud"], index=False)

    print("----- saved submission csv - ", name)
combine_pred_df = combine_predictions(output_file_names)
save_combine_pred_csv(combine_pred_df, 'final_combine_pred_05OCT_2005.csv')