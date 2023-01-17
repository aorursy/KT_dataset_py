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
df = df.append(test)
df.drop('Unnamed: 0',axis=1,inplace=True)
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_',''))
df['codigo_mun']
df = pd.get_dummies( df, columns = ['porte'] )
df['comissionados_por_servidor_limpo'] = df['comissionados_por_servidor'].apply(lambda x: '0' if x == '#DIV/0!' else x)
df['comissionados_por_servidor_novo'] = df['comissionados_por_servidor_limpo'].apply(lambda x: x.replace('%',''))
df['comissionados_por_servidor_novo'] = pd.to_numeric(df['comissionados_por_servidor_novo'])/100
df.isnull().sum()
df.fillna(-1,inplace = True)
for col in df.columns:

    if df[col].dtype == 'object' and col != 'codigo_mun':

       df[col] = df[col].astype('category').cat.codes  
feats = [c for c in df.columns if c not in ['nota_mat','comissionados_por_servidor_limpo','comissionados_por_servidor_limpo','comissionados_por_servidor','codigo_mun']]
feats
df, test = df[df['nota_mat']!=-1], df[df['nota_mat']==-1]
from sklearn.model_selection import train_test_split
train,valid = train_test_split(df, random_state=42, test_size=.1)
df.shape
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(train[feats],train['nota_mat'])
test.shape
valid_preds = rf.predict(valid[feats])
test['nota_mat'] = rf.predict(test[feats])
test[['codigo_mun','nota_mat']].to_csv('rf.csv', index=False)
test[['codigo_mun','nota_mat']]
from sklearn.metrics import mean_squared_error
mean_squared_error(rf.predict(valid[feats]),valid['nota_mat'])**(1/2)
pd.Series(rf.feature_importances_,index=feats).sort_values().plot.barh()