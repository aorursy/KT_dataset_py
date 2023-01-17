import math 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import scipy as sp # scientific computing 
import seaborn as sns # visualization library
import time

from datetime import datetime
from collections import Counter
from subprocess import check_output

import os
print(os.listdir("../input/kkbox-churn-scala-label/"))
print(os.listdir("../input/kkbox-churn-prediction-challenge/"))
df_train_file = "../input/kkbox-churn-scala-label/user_label_201703.csv"
df_train = pd.read_csv(df_train_file, dtype = {'is_churn': 'int8'})
df_train.info()
df_test_file = "../input/kkbox-churn-prediction-challenge/sample_submission_v2.csv"
df_test = pd.read_csv(df_test_file, dtype = {'is_churn': 'int8'})
df_test.info()
df_userlogs_file = "../input/kkbox-churn-prediction-challenge/user_logs.csv"
df_userlogs_file_2 = "../input/kkbox-churn-prediction-challenge/user_logs_v2.csv"
df_userlogs = pd.read_csv(df_userlogs_file, nrows = 36000000)
df_userlogs_2 = pd.read_csv(df_userlogs_file_2)
df_userlogs = df_userlogs.append(df_userlogs_2, ignore_index = True)
df_userlogs.info()
del df_userlogs_2
# group by msno
del df_userlogs['date']
counts = df_userlogs.groupby('msno')['total_secs'].count().reset_index()
# generating new feature 'days_listened'
counts.columns = ['msno', 'days_listened']
sums = df_userlogs.groupby('msno').sum().reset_index()
df_userlogs = sums.merge(counts, how = 'inner', on = 'msno')
# finding avg seconds played per song
# generating new feature 'secs_per_song'
df_userlogs['secs_per_song'] = df_userlogs['total_secs'].div(df_userlogs['num_25'] + df_userlogs['num_50'] + df_userlogs['num_75'] + df_userlogs['num_985'] + df_userlogs['num_100'])
df_userlogs.head()
df_members_file = "../input/kkbox-churn-prediction-challenge/members_v3.csv"
df_members = pd.read_csv(df_members_file)
df_members.info()
# imputing missing values in members dataset
df_members['city'] = df_members.city.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
df_members['bd'] = df_members.bd.apply(lambda x: -99999 if float(x) <= 1 else x )
df_members['bd'] = df_members.bd.apply(lambda x: -99999 if float(x) >= 100 else x )
df_members['gender'] = df_members['gender'].fillna("others")
df_members['registered_via'] = df_members.registered_via.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
current = datetime.strptime('20170331', "%Y%m%d").date()
# generating new feature 'num_days' from 'registration_init_time'
df_members['num_days'] = df_members.registration_init_time.apply(lambda x: (current - datetime.strptime(str(int(x)), "%Y%m%d").date()).days if pd.notnull(x) else "NAN")
del df_members['registration_init_time']
# as city is a heavily skewed, we removed city feature
del df_members['city']
# process of binning
df_members['registered_via'].replace([-1, 1, 2, 5, 6, 8, 10, 11, 13, 14, 16, 17, 18, 19], 1, inplace = True)
# too many outliers so we removed bd feature
del df_members['bd']
# transactions dataset
df_transactions_file = "../input/kkbox-churn-prediction-challenge/transactions.csv"
df_transactions_file_2 = "../input/kkbox-churn-prediction-challenge/transactions_v2.csv"
df_transactions = pd.read_csv(df_transactions_file)
df_transactions_2 = pd.read_csv(df_transactions_file_2)
df_transactions = df_transactions.append(df_transactions_2, ignore_index = True)
df_transactions.describe()
del df_transactions_2
# as payment_method_id is a heavily skewed, we removed payment_method_id feature
#del df_transactions['payment_method_id']
# as payment_plan_days is a heavily skewed, we removed payment_plan_days feature
del df_transactions['payment_plan_days']
# correlation between plan_list_price and actual_amount_paid
df_transactions['plan_list_price'].corr(df_transactions['actual_amount_paid'], method = 'pearson') 
# as highly correlated we removed actual_amount_paid
del df_transactions['actual_amount_paid']
# delete these two columns because is_churn is based on these two features which is already labelled in train dataset
del df_transactions['membership_expire_date']
del df_transactions['transaction_date']
# removing duplicates
df_transactions = df_transactions.drop_duplicates()
df_transactions = df_transactions.groupby('msno').mean().reset_index()
df_transactions.head()
# hot encoding
gender = {'male': 0, 'female': 1, 'others' :2}
# merge the training dataset with members, transaction, userlogs data set
df_training = pd.merge(left = df_train, right = df_members, how = 'left', on = ['msno'])
df_training = pd.merge(left = df_training, right = df_transactions , how = 'left', on = ['msno'])
df_training = pd.merge(left = df_training, right = df_userlogs, how = 'left', on = ['msno'])
df_training['gender'] = df_training['gender'].map(gender)
# merge the testing dataset with members, transaction, userlogs data set
df_testing = pd.merge(left = df_test, right = df_members, how = 'left', on = ['msno'])
df_testing = pd.merge(left = df_testing, right = df_transactions , how = 'left', on = ['msno'])
df_testing = pd.merge(left = df_testing, right = df_userlogs, how = 'left', on = ['msno'])
df_testing['gender'] = df_testing['gender'].map(gender)
del df_members
del df_userlogs
del df_transactions
# Reasons we did not fillna for datasets after merging is because the models later will automatically impute best values for missing values
import sklearn as sl # machine learning
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
cols = [c for c in df_training.columns if c not in ['is_churn','msno']] 
X = df_training[cols] 
Y = df_training['is_churn'] 
validation_size = 0.20
seed = 7
scoring = 'roc_auc'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
# LGBOOST
lgb_params = { 'learning_rate': 0.02, 
               'application': 'binary', 
               'max_depth': 35, 
               'num_leaves': 3500, 
               'verbosity': -1, 
               'metric': 'binary_logloss' 
              } 
d_trainl = lgb.Dataset(X_train, label = Y_train) 
d_validl = lgb.Dataset(X_validation, label = Y_validation) 
watchlistl = [d_trainl, d_validl]
lgb_model = lgb.train(lgb_params, 
                      train_set = d_trainl, 
                      num_boost_round = 1000, 
                      valid_sets = watchlistl, 
                      early_stopping_rounds = 50, 
                      verbose_eval = 10)
lgb_pred = lgb_model.predict(df_testing[cols])
lgb_testing = df_testing.copy()
lgb_testing['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15) 
lgb_testing[['msno','is_churn']].to_csv('lgb_result.csv', index = False)