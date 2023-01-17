# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import feather

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv', index_col=0)

# when using dataframe you can specify some columns that must not be used

not_used = []

cat_feat = []

target = 'target'



# likeli_feat = feather.read_dataframe('./data/likeli_10folds_1000.fth').set_index('ID_code')



# data = pd.merge(data, likeli_feat.iloc[:200000], left_index=True, right_index=True)

features = [i for i in data.columns if i != target]



print(data.shape, len(features))

data.head()
etd = pd.read_csv('../input/test.csv', index_col=0)

# etd = pd.merge(etd, likeli_feat.iloc[200000:], left_index=True, right_index=True)

etd.head()
from tqdm import tqdm



orig = [f'var_{i}' for i in range(200)]

has_one = [f'var_{i}_has_one' for i in range(200)]

has_zero = [f'var_{i}_has_zero' for i in range(200)]

not_u = [f'var_{i}_not_unique' for i in range(200)]



for f in tqdm(orig):

    unique_v = etd[f].value_counts()

    unique_v = unique_v.index[unique_v == 1]

    etd[f + '_u'] = etd[f].isin(unique_v)
etd['has_unique'] = etd[[f + '_u' for f in orig]].any(axis=1)

print(etd['has_unique'].sum())
real_samples = etd.loc[etd['has_unique'], orig]

ref = pd.concat([data, real_samples], axis=0)

print(ref.shape)
for f in tqdm(orig):

    data[f + '_has_one'] = 0

    data[f + '_has_zero'] = 0

    f_1 = data.loc[data[target] == 1, f].value_counts()

    

    f_1_1 = set(f_1.index[f_1 > 1])

    f_0_1 = set(f_1.index[f_1 > 0])



    f_0 = data.loc[data[target] == 0, f].value_counts()

    f_0_0 = set(f_0.index[f_0 > 1])

    f_1_0 = set(f_0.index[f_0 > 0])

    

    data.loc[data[target] == 1, f + '_has_one'] = data.loc[data[target] == 1, f].isin(f_1_1).astype(int)

    data.loc[data[target] == 0, f + '_has_one'] = data.loc[data[target] == 0, f].isin(f_0_1).astype(int)



    data.loc[data[target] == 1, f + '_has_zero'] = data.loc[data[target] == 1, f].isin(f_1_0).astype(int)

    data.loc[data[target] == 0, f + '_has_zero'] = data.loc[data[target] == 0, f].isin(f_0_0).astype(int)



data.loc[:, has_one] = 2*data.loc[:, has_one].values + data.loc[:, has_zero].values
for f in tqdm(orig):

    etd[f + '_has_one'] = 0

    etd[f + '_has_zero'] = 0

    f_1 = data.loc[data[target] == 1, f].unique()

    f_0 = data.loc[data[target] == 0, f].unique()

    etd.loc[:, f + '_has_one'] = etd[f].isin(f_1).astype(int)

    etd.loc[:, f + '_has_zero'] = etd[f].isin(f_0).astype(int)

    

etd.loc[:, has_one] = 2*etd.loc[:, has_one].values + etd.loc[:, has_zero].values
for f in tqdm(orig):

    v = ref[f].value_counts()

    

    non_unique_v = v.index[v != 1]

    

    m_trd = data[f].isin(non_unique_v)

    data[f + '_not_unique'] = m_trd  * data[f] + (~m_trd) * data[f].mean()

    

    m_etd = etd[f].isin(non_unique_v)

    etd[f + '_not_unique'] = m_etd  * etd[f] + (~m_etd) * data[f].mean()

    

    data.loc[~m_trd, f + '_has_one'] = 4

    etd.loc[~m_etd, f + '_has_one'] = 4
data['var_0_has_one'].value_counts()
feather.write_dataframe(data.reset_index(), './921_data.fth')

feather.write_dataframe(etd.reset_index(), './921_etd.fth')

np.save('./real_samples.index', real_samples.index.values)