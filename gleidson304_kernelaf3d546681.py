# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df = df.append(test, sort=False)
df.shape
df.head().T
df.tail().T
df.dtypes
df.info()
test.info()
df.isnull().sum()
df['codigo_mun']= df['codigo_mun'].apply(lambda x:x.replace('ID_ID_',''))

df['codigo_mun']=df['codigo_mun'].values.astype('int64')
df.tail().T
for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes 
df.dtypes
df.min().min()
df.fillna(-2, inplace=True)
test.fillna(-2, inplace=True)
df.isnull().sum()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
df, test = df[df['nota_mat']!=-2], df[df['nota_mat']==-2]
train, valid = train_test_split(df, test_size=0.15, random_state=42)
train, test = train_test_split(df,test_size=0.333, random_state=42)
df.shape, valid.shape, test.shape
rf = RandomForestRegressor(random_state=42, n_estimators=100)
feats = [c for c in df.columns if c not in ['nota_mat']]
rf.fit(df[feats], df['nota_mat'])
for col in test.columns:

    if test[col].dtype == 'object':

        test[col] = test[col].astype('category').cat.codes
from sklearn.metrics import mean_squared_error
valid_preds = rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
test['nota_mat'] = rf.predict(test[feats])
test[['nota_mat','codigo_mun']].to_csv('rf.csv', index=False)