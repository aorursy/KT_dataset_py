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
teste = pd.read_csv('../input/test.csv')
df.info()
df.shape, teste.shape
df = pd.concat([df, teste], axis=0)
for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df.loc[2]
df.reset_index(inplace=True)
df.head()
df.fillna(value=-1, inplace=True)
df['nota_mat'] = np.where(df['nota_mat']==-1,np.nan, df['nota_mat'])
teste = df[df['nota_mat'].isnull()]

df = df[~df['nota_mat'].isnull()]
df.shape, teste.shape
from sklearn.model_selection import train_test_split
treino, validacao = train_test_split(df, test_size=0.20, random_state=43)
treino.shape, validacao.shape, teste.shape
from sklearn.ensemble import RandomForestClassifier
removed_cols = ['index', 'municipio', 'nota_mat','capital','densidade_dem','exp_anos_estudo','populacao']

feats = [c for c in df.columns if c not in removed_cols]
randomForest = RandomForestClassifier(random_state=43)
randomForest.fit(treino[feats], treino['nota_mat'])
predito = randomForest.predict(validacao[feats])
from sklearn.metrics import accuracy_score
accuracy_score(validacao['nota_mat'], predito)
df['nota_mat']
feats
predito
randomForest.predict(teste[feats])
teste['nota_mat'] = randomForest.predict(teste[feats])

teste[['codigo_mun', 'nota_mat']].to_csv('randomForest_isaias.csv', index=False)
teste[['codigo_mun', 'nota_mat']]