# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import xgboost as xgb

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/train.csv')

macro_df=pd.read_csv('../input/macro.csv')

test_df=pd.read_csv('../input/test.csv')

train_df.shape
macro_df.shape
test_df.shape
inv_train=train_df[train_df['product_type']=='Investment']
inv_train.head()
inv_mac_train=pd.concat([inv_train,macro_df],axis=0)
inv_test=test_df[test_df['product_type']=='Investment']
inv_test.head()
inv_mac_test=pd.concat([inv_test,macro_df],axis=0)
inv_mac_test

inv_mac_train
corrmat = inv_mac_train.corr()

n = 20

cols = corrmat.nlargest(n, 'price_doc')['price_doc'].index

cm_df = inv_mac_train[cols].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(cm_df, square=True, annot=True, fmt='.2f', annot_kws={'size':10}, cbar=True)
import seaborn as sns

import matplotlib.pyplot as plt
o_train=train_df[train_df['product_type']=='OwnerOccupier']
o_mac_train=pd.concat([o_train,macro_df],axis=0)
corrmat = o_mac_train.corr()

n = 20

cols = corrmat.nlargest(n, 'price_doc')['price_doc'].index

cm_df = o_mac_train[cols].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(cm_df, square=True, annot=True, fmt='.2f', annot_kws={'size':10}, cbar=True)
o_test=test_df[test_df['product_type']=='OwnerOccupier']
o_mac_test=pd.concat([o_test,macro_df],axis=0)
df_train = inv_train

df_test = inv_test

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
df_train.head()
print(df_train.shape)

df_train.loc[df_train.full_sq == 0, 'full_sq'] = 30

df_train = df_train[df_train.price_doc/df_train.full_sq <= 600000]

df_train = df_train[df_train.price_doc/df_train.full_sq >= 10000]

print(df_train.shape)
y_train = df_train['price_doc'].values

id_test =test_df['id']
df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = df_all.join(macro_df, on='timestamp', rsuffix='_macro')

print(df_all.shape)
df_all