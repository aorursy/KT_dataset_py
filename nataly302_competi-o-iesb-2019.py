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
import numpy as np

import pandas as pd
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df.head()
test.head()
df.shape
df.head().T
df.dtypes
df.info()
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_','')) 

test['codigo_mun'] = test['codigo_mun'].apply(lambda x: x.replace('ID_ID_','')) 
df = df.apply(lambda x:x.replace('.' , ','))
df['codigo_mun'] = df['codigo_mun'].values.astype('int64')

test['codigo_mun'] = test['codigo_mun'].values.astype('int64')
# Preenchendo os valores faltantes com -1

# ver se tem nulos: df.isnull().sum()

df.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)
# Codificando texto em número

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
from sklearn.model_selection import train_test_split
train ,valid = train_test_split(df,  random_state=42)



for col in test.columns:

    if test[col].dtype == 'object':

       test[col] = test[col].astype('category').cat.codes
# Separando as features

feats = [c for c in df.columns if c not in ['nota_mat']]
from sklearn.ensemble import RandomForestRegressor
# Instanciando a RandomForest



rf = RandomForestRegressor( random_state=42 )
# Treinando o modelo

rf.fit(train[feats], train['nota_mat'])
from sklearn.metrics import mean_squared_error



mean_squared_error(valid['nota_mat'], rf.predict(valid[feats]))**1/2
(valid['nota_mat'] == rf.predict(valid[feats])).mean()
valid_preds = rf.predict(valid[feats])
##mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
test['nota_mat'] = (rf.predict(test[feats]))
test[['codigo_mun' , 'nota_mat']].to_csv('Natalia.csv', index=False)