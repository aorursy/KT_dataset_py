import numpy as np

import pandas as pd

from sklearn import metrics, preprocessing, model_selection

import lightgbm as lgb

import matplotlib.pyplot as plt



%matplotlib inline



pd.options.display.max_columns = 100
!ls ../input/
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test_bqCt9Pv.csv")

print(train_df.shape, test_df.shape)
train_df.head()
train_df["loan_default"].value_counts()