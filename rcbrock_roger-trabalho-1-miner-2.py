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
df.head()
test = pd.read_csv('../input/test.csv')
test.head()
df.info()
df.shape, test.shape
df = pd.concat([df, test], axis=0)
for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df
df.dtypes
df.shape
df.head()
df
df.loc[2]
df.reset_index(inplace=True)
df.head()
df.fillna(value=-1, inplace=True)
df.head()
df['nota_mat'] = np.where(df['nota_mat']==-1,np.nan, df['nota_mat'])
df
test = df[df['nota_mat'].isnull()]

df = df[~df['nota_mat'].isnull()]
df.shape, test.shape
from sklearn.model_selection import train_test_split



train, valid = train_test_split(df, test_size=0.20, random_state=42)
train.shape, valid.shape, test.shape
test.dtypes
removed_cols = ['index', 'municipio', 'nota_mat','capital','densidade_dem','exp_anos_estudo','populacao']

feats = [c for c in df.columns if c not in removed_cols]
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(train[feats], train['nota_mat'])
preds = rf.predict(valid[feats])
from sklearn.metrics import accuracy_score
accuracy_score(valid['nota_mat'], preds)
df['nota_mat']
feats
preds
test['nota_mat'] = rf.predict(test[feats])

test[['codigo_mun', 'nota_mat']].to_csv('rf9.csv', index=False)