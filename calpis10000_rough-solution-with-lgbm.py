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
import pandas as pd

import numpy as np

import scipy



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GroupKFold

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

import lightgbm as lgb



import gc

import zipfile
train = pd.read_csv('../input/mercari-unzipped/train.tsv', sep='\t')

test = pd.read_csv('../input/mercari-unzipped/test.tsv', sep='\t')
train.head()
test.head()
# https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/jn/EDA.py

def df_info(target_df, topN=10):

    

    max_row = target_df.shape[0]

    print(f'Shape: {target_df.shape}')

    

    df = target_df.dtypes.to_frame()

    df.columns = ['DataType']

    df['#Nulls'] = target_df.isnull().sum()

    df['#Uniques'] = target_df.nunique()

    

    # stats

    df['Min']   = target_df.min(numeric_only=True)

    df['Mean']  = target_df.mean(numeric_only=True)

    df['Max']   = target_df.max(numeric_only=True)

    df['Std']   = target_df.std(numeric_only=True)

    

    # top 10 values

    df[f'top{topN} val'] = 0

    df[f'top{topN} cnt'] = 0

    df[f'top{topN} raito'] = 0

    for c in df.index:

        vc = target_df[c].value_counts().head(topN)

        val = list(vc.index)

        cnt = list(vc.values)

        raito = list((vc.values / max_row).round(2))

        df.loc[c, f'top{topN} val'] = str(val)

        df.loc[c, f'top{topN} cnt'] = str(cnt)

        df.loc[c, f'top{topN} raito'] = str(raito)

        

    return df



df_info(train)
df_info(test)
# 雑な可視化。カテゴリっぽいカラムと目的変数の相関だけ観察。

train[['item_condition_id', 'price']].groupby('item_condition_id').sum()
train[['category_name', 'price']].groupby('category_name').sum().sort_values('price', ascending=False).head()
train[['category_name', 'price']].groupby('category_name').sum().sort_values('price', ascending=True).head()
train[['brand_name', 'price']].groupby('brand_name').sum().sort_values('price', ascending=False).head()
train[['brand_name', 'price']].groupby('brand_name').sum().sort_values('price', ascending=True).head()
# 文字列の情報を使わずに、trainデータだけで超雑なモデルを作ってみる。

tgt_cols_x = ['item_condition_id', 'category_name', 'brand_name', 'shipping']

tgt_cols_y = ['price']

train_v01 = train.set_index('train_id').copy()[tgt_cols_x + tgt_cols_y]

train_v01['brand_name'] = train_v01['brand_name'].fillna('NonBrand')

X = train_v01[tgt_cols_x].astype('category')

y = train_v01[tgt_cols_y]
X.head()
y.head()
params = {

    'learning_rate': 0.75,

    'application': 'regression',

    'max_depth': 3,

    'num_leaves': 100,

    'verbosity': -1,

    'metric': 'RMSE',

}
skf = StratifiedKFold(n_splits=5)

skf.get_n_splits(X, y)
print(skf)
for train_index, test_index in skf.split(X, y):

    print(train_index)
from sklearn.model_selection import KFold



skf = KFold(n_splits=5)



for train_index, test_index in skf.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]

    y_train, y_test = y.loc[train_index,:], y.loc[test_index,:]

    model = lgb.train(params, train_set=d_train, num_boost_round=3200, verbose_eval=100) 