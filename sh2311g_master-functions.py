# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_rows',200)
pd.set_option('max_columns',200)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

# drop columns
def dropify(cols, df):
    df = df.drop(cols, axis=1)
    return df

#label encoding
def label_encode(train, test, col_list):
    le=LabelEncoder()
    for col in col_list:
        # Using whole data to form an exhaustive list of levels
        data=train[col].append(test[col])
        le.fit(data.values.astype(str))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    return (train, test)
# one hot
def one_hot(df, col, drop=True):
    oh = pd.get_dummies(df[col], prefix=col)
    if drop:
        df = df.drop(col, axis=1)
    df = df.join(oh)
    return df
# confusion matrix
def confusion_matric_accuracy(cm):
    return np.trace(cm)/np.sum(cm)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.5, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train_labels = le.transform(y_train)
y_train_kerased = to_categorical(y_train_labels)