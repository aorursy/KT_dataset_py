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



import matplotlib.pyplot as plt

import seaborn as sns
# Importação dos dados

df_treino = pd.read_csv('../input/train.csv', parse_dates=[0])

df_teste = pd.read_csv('../input/test.csv', parse_dates=[0])
# União dos arquivos

df = df_treino.append(df_teste, sort=False)
df.info()
# Transformado valores



# Area em float

df['area'] = df['area'].str.replace(',','')

df['area'] = df['area'].astype('float64') 



# Densidade em float

df['densidade_dem'] = df['densidade_dem'].str.replace(',','')

df['densidade_dem'] = df['densidade_dem'].astype('float64') 



# Servidores comissionados

df['servidores'] = df['comissionados'].fillna(df['comissionados'])



# Comissionados por servidor

df['comissionados_por_servidor'].replace({'%': ''}, inplace=True, regex=True)

df.comissionados_por_servidor = pd.to_numeric(df.populacao, errors='coerce').fillna(-1).astype(np.int64)



# Transformando população em inteiro

df['populacao'].replace({'\(1\)': '', '\(2\)': '', '\(3\)': '', '[.]': '', ',': ''}, inplace=True, regex=True)

df.populacao = pd.to_numeric(df.populacao, errors='coerce').fillna(-1).astype(np.int64)
df['populacao'].describe().apply(lambda x: format(x, 'f'))
plt.figure(figsize=(12,6))

sns.distplot(df['populacao'])

plt.xlabel('População')

plt.ylabel("Quantidade de Cidades")

plt.title("Distribuição da População pelas Cidades dos filmes")
# Transformando dummies

dummy_regiao = pd.get_dummies(df['regiao'])

df = pd.concat([df, dummy_regiao], axis = 1)



dummy_estado = pd.get_dummies(df['estado'])

df = pd.concat([df, dummy_estado], axis = 1)



dummy_porte = pd.get_dummies(df['porte'])

df = pd.concat([df, dummy_porte], axis = 1)
# Remover e preparar colunas

removed_cols = ['municipio', 'codigo_mun', 'regiao', 'estado', 'porte', 'nota_mat', 'capital']

feats = [c for c in df.columns if c not in removed_cols]
df.head()

train = df[~df['nota_mat'].isnull()]

train = train.fillna(-1)

test = df[df['nota_mat'].isnull()]

test = test.fillna(-1)
train.head()
from sklearn.model_selection import train_test_split

train_split, valid = train_test_split(train, random_state=666)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=666, n_jobs=-1, n_estimators=200)
rf.fit(train_split[feats], train_split['nota_mat'])

preds = rf.predict(valid[feats])
from sklearn.metrics import accuracy_score

accuracy_score(valid['nota_mat'], preds)
# Aplica o modelo final usando toda a base como treino

rf.fit(train[feats], train['nota_mat'])

test['nota_mat'] = rf.predict(test[feats])
test[['codigo_mun', 'nota_mat']].to_csv('rf1.csv', index=False)