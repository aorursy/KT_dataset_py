import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Libraries

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 50)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import datetime

import missingno as msno

from matplotlib import dates



from tqdm import tqdm



import warnings

warnings.filterwarnings('ignore')

import gc



import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("../input/tmu-inclass-competition/train.csv")

test_df = pd.read_csv("../input/tmu-inclass-competition/test.csv")

sample_df = pd.read_csv("../input/tmu-inclass-competition/sample_submission.csv")
print("学習データのサンプル数は{}".format(len(train_df)))

print("特徴量の数はtargetを含めて{}".format(test_df.shape[1]))

train_df.head()
print("テストデータのサンプル数は{}".format(len(test_df)))

test_df.head()
sample_df.head()
msno.matrix(train_df);
msno.matrix(test_df);
tmp = pd.DataFrame(columns=["train", "test"])

tmp["train"] = train_df.isnull().sum().sort_values(ascending=False)[:15]

tmp["test"] = test_df.isnull().sum().sort_values(ascending=False)[:15]

print("欠損値の数リスト")

tmp
fig, ax = plt.subplots()

sns.distplot(train_df["price"], ax=ax, kde=False)

ax.set_ylabel("count");
train_df["price"].describe()
train_df["host_id"].value_counts()[:20]
test_df["host_id"].value_counts()[:20]
train_df["host_is_superhost"].value_counts()
train_df.loc[train_df["host_is_superhost"] == "f", "price"].mean()
train_df.loc[train_df["host_is_superhost"] == "t", "price"].mean()
corr = train_df.corr()

fig, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corr, annot=True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            square=True,

            ax=ax);
train_df.columns

train_df.loc[:, ["accommodates", 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'square_feet']].describe()