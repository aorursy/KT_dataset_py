# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import lightgbm as lgb

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("../input/ds2019uec-task1/train.csv")

test_df = pd.read_csv("../input/ds2019uec-task1/test.csv")
# train_df.head()
# test_df.head()
train = train_df.copy()

test = test_df.copy()
encoders = dict()

for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:

    le = preprocessing.LabelEncoder()

    le.fit(pd.concat([train_df[col], test_df[col]], axis=0))

    train[col] = le.transform(train_df[col])

    test[col] = le.transform(test_df[col])

    encoders[col] = le
train["y"] = (train_df["y"] == "yes").astype(np.int)
# train.head()
train_data = lgb.Dataset(train.drop("y", axis=1), label=train["y"])
param = {'num_leaves': 31, 'objective': 'binary'}

param['verbose'] = -1

param['metric'] = 'auc'
cv_result = lgb.cv(param, train_data, 50, nfold=5, verbose_eval=False)
np.array(cv_result["auc-mean"]).argmax()
bst = lgb.train(param, train_data, 42)
# lgb.plot_importance(bst, figsize=(12, 6))
ypred = bst.predict(test)
sub = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

sub["pred"] = ypred

sub.to_csv("sample_lgbm.csv", index=False)
# train_y_pred = bst.predict(train.drop("y", axis=1))

# plt.hist(train_y_pred)
# plt.hist(ypred)