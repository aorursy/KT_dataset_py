# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import roc_curve, roc_auc_score

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

df = pd.read_csv("/kaggle/input/uts-2020-autumn-data-analytics-assignment-3/train.csv", index_col="ID")

df_test = pd.read_csv("/kaggle/input/uts-2020-autumn-data-analytics-assignment-3/test-pub.csv", index_col="ID")

df_onehot = pd.get_dummies(df)



keys = df_onehot.keys()

data_keys = [k for k in keys 

    if '?' not in k and k[-3:] != "50K"]

data_train = df_onehot[data_keys]

target_train = df_onehot["Salary_ >50K"]  



df_onehot1 = pd.get_dummies(df_test)

# add all zero to non-existing keys

for k in data_keys:

    if k not in df_onehot1.keys():

        df_onehot1[k] = 0



data_test = df_onehot1[data_keys]
import sklearn.preprocessing as prep

from sklearn.linear_model import LogisticRegression

sc = prep.MinMaxScaler()

data_train_s = sc.fit_transform(data_train)

data_test_s = sc.transform(data_test)



lr = LogisticRegression()

lr.fit(data_train_s, target_train)

# Predict the probability of postive class

pred_test_prob = lr.predict_proba(data_test_s)[:, 1] # 



df_test["Predicted"] = pred_test_prob

df_test["Predicted"].to_csv("/kaggle/working/LogisticReg_v0.csv")