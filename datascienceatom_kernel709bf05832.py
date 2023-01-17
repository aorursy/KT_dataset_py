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
#01. Importar Bibliotecas



import numpy as np

import pandas as pd

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import matplotlib

from sklearn.metrics import accuracy_score

#02. Abrir arquivos



df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#03. Transformar a coluna nota_mat em log

df['nota_mat'] = np.log(df['nota_mat'])
#04. Juntar os 2 DataFrames em uma única variável



df = df.append(test, sort=False)

               
df.shape
#05. Transformar texto em número, exceto a codigo_num

for c in df.columns:

    if (df[c].dtype == 'object') & (c != 'codigo_mun'):

        df[c] = df[c].astype('category').cat.codes
#06. Preencher Valores em Branco

df['codigo_mun'] = df['codigo_mun'].str.replace('ID_ID_','')
df['nota_mat'].fillna(-2, inplace=True)

df.fillna(0, inplace=True)
#07. Separar as bases de volta em test e df



df, test = df[df['nota_mat']!=-2], df[df['nota_mat']==-2]
#08. Criar os conjuntos de treino e validação

train, valid = train_test_split(df, random_state=42)

train, valid = train_test_split(df, random_state=42)

train, valid = train_test_split(df, random_state=42)
#09. Treinar o modelo

rf = RandomForestRegressor(random_state=42, n_estimators=100)
feats = [c for c in df.columns if c not in ['nota_mat']]
rf.fit(train[feats], train['nota_mat'])
#10. Fazer previsões

valid_preds = rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
#Enviar pro Kaggle

test['nota_mat'] = np.exp(rf.predict(test[feats]))
test[['codigo_mun','nota_mat']].to_csv('rf1.csv', index=False)