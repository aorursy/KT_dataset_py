import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as K
import tensorflow as tf
import gzip
import random
%matplotlib inline
random.seed(5566)
num_lines = 40428968 - 1# train.gz file lines counting
sample_size = 2000000
skip = sorted(random.sample(range(1, num_lines), num_lines-sample_size))
skip[0]
# skip = skip[1:] # if contains index=0, the header will be removed
train_file_path = "../input/avazu-ctr-prediction/train.gz"

# skip = sorted(random.sample(range(num_lines), num_lines-sample_size))
train_df = pd.read_csv(
    train_file_path, 
    header=0,
    skiprows=skip)
print(len(train_df))
train_df.head()
test_df = pd.read_csv("../input/avazu-ctr-prediction/test.gz", header=0)
submission_df = pd.read_csv("../input/avazu-ctr-prediction/sampleSubmission.gz")
def to_date_column(df):
    df["dt_hour"] = pd.to_datetime(df["hour"], format="%y%m%d%H")
    df["year"] = df["dt_hour"].dt.year
    df["month"] = df["dt_hour"].dt.month
    df["day"] = df["dt_hour"].dt.day
    df["int_hour"] = df["dt_hour"].dt.hour
    df["is_weekday"] = df["dt_hour"].dt.dayofweek
    df["is_weekend"] = df.apply(lambda x: x["is_weekday"] in [5, 6], axis=1)
to_date_column(train_df)
train_df.head()
train_df.columns
train_df.nunique()
label_col = "click"
x_columns = set(list(train_df.columns)) - set(["id", "site_id", "app_id", "hour", "dt_hour", "device_id", "device_ip"] + [label_col])
# x_columns
x_train = train_df[x_columns]
y_train = train_df[label_col]
# x_train.head()
x_train.nunique()
to_date_column(test_df)
x_test = test_df[x_columns]
# test_df.head()
del train_df
del test_df
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


x_train_len = len(x_train)
n_df = pd.concat([x_train, x_test])
d = defaultdict(LabelEncoder)
n_df = n_df.apply(lambda x: d[x.name].fit_transform(x))
n_df.head()
# n_df = pd.get_dummies(n_df)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaler.fit(n_df)
n_df = scaler.transform(n_df)
len(n_df)
n_x_train = n_df[:x_train_len]
n_x_test = n_df[x_train_len:]
from sklearn.linear_model import LogisticRegression


assert len(n_x_train) == len(y_train)
clf = LogisticRegression(verbose=True)
clf.fit(n_x_train, y_train)


y_train_star = clf.predict(n_x_train)
from sklearn.metrics import log_loss


log_loss(y_train, y_train_star)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

accuracy_score(y_train, y_train_star)
confusion_matrix(y_train, y_train_star)
y_test = clf.predict(n_x_test)
y_test = clf.predict_proba(n_x_test)
print(clf.classes_)
print(y_test.shape)
print(y_test[3:5, ])

y_test = y_test[:, 1]
print(y_test.shape)
print(y_test[3:5])
submission_df["click"] = y_test
# y_pred_proba.to_csv('submission.csv', index=False)
submission_df.to_csv("submission.csv", index=False)
submission_df.head()