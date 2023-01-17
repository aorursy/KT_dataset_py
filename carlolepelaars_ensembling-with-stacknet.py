# Import PyStackNet Package

# Source: https://www.kaggle.com/kirankunapuli/pystacknet

import os

import sys

sys.path.append("../input/pystacknet/repository/h2oai-pystacknet-af571e0")

import pystacknet
# Standard Dependencies

import gc

import sys

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve



# Machine Learning

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



# Specify paths and key features

KAGGLE_DIR = '../input/ieee-fraud-detection/'

TARGET = 'isFraud'



# Seed for reproducability

seed = 1234

np.random.seed(seed)



# For keeping time. Limit for Kaggle kernels is set to approx. 9 hours.

t_start = time.time()
# File sizes and specifications

print('\n# Files and file sizes:')

for file in os.listdir(KAGGLE_DIR):

    print('{}| {} MB'.format(file.ljust(30), 

                             str(round(os.path.getsize(KAGGLE_DIR + file) / 1000000, 2))))
def auc_score(y_true, y_pred):

    """

    Calculates the Area Under ROC Curve (AUC)

    """

    return roc_auc_score(y_true, y_pred)
def plot_curve(y_true_train, y_pred_train, y_true_val, y_pred_val, model_name):

    """

    Plots the ROC Curve given predictions and labels

    """

    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_pred_train, pos_label=1)

    fpr_val, tpr_val, _ = roc_curve(y_true_val, y_pred_val, pos_label=1)

    plt.figure(figsize=(8, 8))

    plt.plot(fpr_train, tpr_train, color='black',

             lw=2, label=f"ROC train curve (AUC = {round(roc_auc_score(y_true_train, y_pred_train), 4)})")

    plt.plot(fpr_val, tpr_val, color='darkorange',

             lw=2, label=f"ROC validation curve (AUC = {round(roc_auc_score(y_true_val, y_pred_val), 4)})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.xticks(fontsize=14)

    plt.yticks(fontsize=14)

    plt.title(f'ROC Plot for {model_name}', weight="bold", fontsize=20)

    plt.legend(loc="lower right", fontsize=16)
# Load in datasets

train_transaction = pd.read_csv(f"{KAGGLE_DIR}train_transaction.csv", index_col='TransactionID')

test_transaction = pd.read_csv(f"{KAGGLE_DIR}test_transaction.csv", index_col='TransactionID')

train_identity = pd.read_csv(f"{KAGGLE_DIR}train_identity.csv", index_col='TransactionID')

test_identity = pd.read_csv(f"{KAGGLE_DIR}test_identity.csv", index_col='TransactionID')



# Merge datasets into full training and test dataframe

train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True).reset_index()

test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True).reset_index()

del train_identity, train_transaction, test_identity, test_transaction

gc.collect()
# Select only first 52 features

# The other columns are quite noisy

train_df = train_df.iloc[:, :53]

test_df = test_df.iloc[:, :52]
def reduce_mem_usage(df):

    """

    Reduces memory usage for all columns in a Pandas DataFrame

    """

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings                       

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int32)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    else:

                        df[col] = df[col].astype(np.uint32)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    else:

                        df[col] = df[col].astype(np.int32)

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)



    # Print final result

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is after reduction is:",mem_usg," MB")

    return df, NAlist
# Reduce memory

train_df, _ = reduce_mem_usage(train_df)

test_df, _ = reduce_mem_usage(test_df)
# Drop nuisance columns and specify target variable

X_train = train_df.drop([TARGET, 'TransactionID', 'TransactionDT'], axis=1)

X_test = test_df.drop(['TransactionID', 'TransactionDT'], axis=1)

target = train_df[TARGET]



# Label Encoding

lbl = LabelEncoder()

for f in X_train.columns:

    if X_train[f].dtype == 'object': 

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))   
# Split Train and Validation

X_train, X_val, y_train, y_val = train_test_split(X_train,

                                                  target,

                                                  test_size=0.15, 

                                                  random_state=seed, 

                                                  stratify=target)
# Level 1 are the base models that take the training dataset as input

l1_clf1 = GradientBoostingRegressor(n_estimators=400,

                                    learning_rate=0.006,

                                    min_samples_leaf=10,

                                    max_depth=15, 

                                    max_features='sqrt', 

                                    subsample=0.85,

                                    random_state=seed)



l1_clf2 = LGBMRegressor(boosting_type='gbdt',

                        objective="binary",

                        metric="AUC",

                        boost_from_average="false",

                        learning_rate=0.0045,

                        num_leaves=491,

                        max_depth=20,

                        min_child_weight=0.035,

                        feature_fraction=0.38,

                        bagging_fraction=0.42,

                        min_data_in_leaf=100,

                        max_bin=255,

                        importance_type='split',

                        reg_alpha=0.4,

                        reg_lambda=0.65,

                        bagging_seed=seed,

                        random_state=seed,

                        verbosity=-1,

                        subsample=0.85,

                        colsample_bytree=0.8,

                        min_child_samples=79)



l1_clf3 = CatBoostRegressor(learning_rate=0.2,

                            bagging_temperature=0.1, 

                            l2_leaf_reg=30,

                            depth=12, 

                            max_bin=255,

                            iterations=100,

                            loss_function='Logloss',

                            objective='RMSE',

                            eval_metric="AUC",

                            bootstrap_type='Bayesian',

                            random_seed=seed,

                            early_stopping_rounds=10)



# Level 2 models will take predictions from level 1 models as input

# Remember to keep level 2 models smaller

# Basic models like Ridge Regression with large regularization or small random forests work well

l2_clf1 = RandomForestRegressor(n_estimators=250, 

                                max_depth=5, 

                                max_features='sqrt', 

                                random_state=seed)
# Specify model tree for StackNet

models = [[l1_clf1, l1_clf2, l1_clf3], # Level 1

          [l2_clf1]] # Level 2
from pystacknet.pystacknet import StackNetClassifier



# Specify parameters for stacked model and begin training

model = StackNetClassifier(models, 

                           metric="auc", 

                           folds=3,

                           restacking=False,

                           use_retraining=True,

                           use_proba=True, # To use predict_proba after training

                           random_state=seed,

                           n_jobs=-1, 

                           verbose=1)



# Fit the entire model tree

model.fit(X_train, y_train)
# Get score on training set and validation set for our StackNetClassifier

train_preds = model.predict_proba(X_train)[:, 1]

val_preds = model.predict_proba(X_val)[:, 1]

train_score = auc_score(y_train, train_preds)

val_score = auc_score(y_val, val_preds)
print(f"StackNet AUC on training set: {round(train_score, 4)}")

print(f"StackNet AUC on validation set: {round(val_score, 4)}")
# Plot ROC curve

plot_curve(y_train, train_preds, y_val, val_preds, "StackNet Baseline")
# Write predictions to csv

sub = pd.read_csv(f"{KAGGLE_DIR}sample_submission.csv")

preds = model.predict_proba(X_test)[:, 1]

sub[TARGET] = preds

sub.to_csv(f"submission.csv", index=False)
# Check Submission format

print("Final Submission Format:")

sub.head()
plt.figure(figsize=(12,4))

plt.hist(train_df[TARGET], bins=100)

plt.title("Distribution for train set", weight='bold', fontsize=18)

plt.xlabel("Predictions", fontsize=15)

plt.ylabel("Frequency", fontsize=15)

plt.xlim(0, 1);
plt.figure(figsize=(12,4))

plt.hist(sub[TARGET], bins=100)

plt.title("Prediction Distribution for test set", weight='bold', fontsize=18)

plt.xlabel("Predictions", fontsize=15)

plt.ylabel("Frequency", fontsize=15)

plt.xlim(0, 1);
# Check kernels run-time. Limit for Kaggle Kernels is set to approx. 9 hours.

t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)

print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 

                                                      int(total_time*60)))