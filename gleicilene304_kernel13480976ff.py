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
df=pd.read_csv("../input/train.csv")

test_X=pd.read_csv("../input/test.csv")
df_raw=df.copy()
df.head().T
df.tail().T
df.dtypes
df.info()
import pandas as pd
df.groupby('regiao')['nota_mat'].mean().plot(kind='barh')

df.groupby('municipio')['nota_mat'].mean()
df['nota_mat'].describe()
df['nota_mat'].hist()
df['codigo_mun']= df['codigo_mun'].apply(lambda x:x.replace('ID_ID_',''))

df['codigo_mun']=df['codigo_mun'].values.astype('int64')
df['regiao'] = df['regiao'].astype('category').cat.codes

df['estado'] = df['estado'].astype('category').cat.codes

df['municipio'] = df['municipio'].astype('category').cat.codes

df['porte'] = df['porte'].astype('category').cat.codes

df['area'] = df['area'].astype('category').cat.codes

df['densidade_dem'] = df['densidade_dem'].astype('category').cat.codes

df['ranking_igm'] = df['ranking_igm'].astype('category').cat.codes

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].astype('category').cat.codes

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.15, random_state=42)
train, valid = train_test_split(df, test_size=0.333, random_state=42)
df.shape,test_X.shape,valid.shape,
feats = [c for c in df.columns if c not in ['nota_mat']]
from sklearn.ensemble import RandomForestRegressor
rf1 = RandomForestRegressor(n_estimators=200, min_samples_split=6, max_depth=10, random_state=42)
df.fillna(-1, inplace=True)
test_X.fillna(-1,inplace=True)
valid.fillna(-1,inplace=True)
rf1.fit(df[feats], df['nota_mat'])
test_X['codigo_mun']= test_X['codigo_mun'].apply(lambda x:x.replace('ID_ID_',''))

test_X['codigo_mun']=test_X['codigo_mun'].values.astype('int64')
for col in test_X.columns:

    if test_X[col].dtype == 'object':

        test_X[col] = test_X[col].astype('category').cat.codes   
test_X.tail()
valid_preds = rf1.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
x=df[feats]

y=df['nota_mat']
rf1.score(x,y)
(valid['nota_mat'] == rf1.predict(valid[feats])).mean()
test_X['nota_mat'] = rf1.predict(test_X[feats])
test_X[['codigo_mun','nota_mat']].to_csv('rf1.csv', index=False)