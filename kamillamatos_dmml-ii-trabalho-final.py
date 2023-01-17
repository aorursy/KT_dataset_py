# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
#Importando as Bibliotecas

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

%matplotlib inline
#Verificando o carregamento da base

df.shape
#Variáveis e dicionário de dados

VAR = [["BAD", "1 representa mal pagador e 0 para bom pagador"],

             ["LOAN", "Montante da solicitação de empréstimo"],

             ["MORTDUE", "Valor devido da hipoteca existente"],

             ["VALUE", "Valor da propriedade atual"],

             ["REASON", "Razão da Consolidação da dívida/Melhoramento da casa"],

             ["JOB", "Seis categorias profissionais"],

             ["YOJ", "Anos no emprego atual"],

             ["DEROG", "Número de principais relatórios depreciativos"],

             ["DELINQ", "Número de linhas de crédito inadimplentes"],

             ["CLAGE", "Idade da linha comercial mais antiga em meses"],

             ["NINQ", "Número de linhas de crédito recentes"],

             ["CLNO", "Número de linhas de crédito"],

             ["DEBTINC", "Razão da dívida / rendimento"]]

            

DF_VAR = pd.DataFrame(VAR, columns=["Variavel", "Descrição"])

DF_VAR

# Verificando os tipos e tamanhos dos dados

df.info()
# Conferindo amostra dos dados

df.sample(5).T
df.update(df['JOB'].fillna('Other'))

df.head(20).T
df = df.dropna(subset=['LOAN'])

df.head(20).T
df.shape
df.fillna(0, inplace=True)
df.describe()
df.nlargest(5, 'LOAN')[['REASON', 'LOAN']].style.hide_index()
df.nlargest(5, 'MORTDUE')[['MORTDUE']].style.hide_index()
plt.figure(figsize=(30,6))

plt.subplot(1,2,1)

fig = df.LOAN.hist(bins=25)

fig.set_title('Distribuição de Valores Empréstimos (LOAN)')

fig.set_xlabel('Valores de empréstimos')

fig.set_ylabel('Quantidade de empréstimos')
plt.figure(figsize=(30,6))

plt.subplot(1,2,1)

fig = df.MORTDUE.hist(bins=25)

fig.set_title('Distribuição de Valores Hipotacas (MORTDUE)')

fig.set_xlabel('Valores de hipotecas')

fig.set_ylabel('Quantidade de hipotecas')
jobs = df['JOB'].dropna().unique()

plt.figure(figsize=(18,19))

c=1

for i in jobs:

    plt.subplot(7,1,c)

    plt.title(i)

    df[df['JOB'] == i]['VALUE'].hist(bins=20)

    c+=1

plt.tight_layout() 
f, ax = plt.subplots(figsize=(25,12))

sns.heatmap(df.corr(), annot=True, fmt='.2f', linecolor='blue', ax=ax, lw=.7)
# Dividir a base em treino e teste

train, test = train_test_split(df, test_size=0.20, random_state=42)



# verificando tamanhos

train.shape, test.shape
# Lista das colunas nao usadas

removed_cols = ['BAD','REASON','JOB']



# Criar a lista das colunas de entrada

feats = [c for c in train.columns if c not in removed_cols]
# XGBoost

# Instanciar o modelo

xgb = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42, learning_rate=0.05)
scores = cross_val_score(xgb, train[feats], train['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()
xgb.fit(train[feats], train['BAD'])
# Fazendo predições

preds = xgb.predict(test[feats])

preds
preds
# Verificando o real

test['BAD'].head(3)
# Medir o desempenho do modelo XGB

accuracy_score(test['BAD'], preds)
# Instanciar o modelo

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
#Treinando o modelo RandomForest

rf.fit(train[feats], train['BAD'])
# Fazendo previsões em cima dos dados de teste da árvore

predstree = rf.predict(test[feats])
# Verificando as previsoes

predstree
# Aplicando a metrica

mean_squared_error(test['BAD'], predstree)**(1/2)
# Conhecendo os valores da variável JOB

df.JOB.unique()
# Conhecendo os valores da variável REASON

df.REASON.unique()
# Criando dummys para as variáveis do tipo "objeto" (JOB e REASON)

df = pd.get_dummies(df, columns=['JOB','REASON'], dtype=int)
#Novo tamanho do dataset

df.shape
# Dividir a base em treino e validação

train, test = train_test_split(df, random_state=42)
# verificando tamanhos

train.shape, test.shape
# Lista das colunas nao usadas

removed_cols = ['BAD']



# Criar a lista das colunas de entrada

feats = [c for c in train.columns if c not in removed_cols]
# XGBoost

# Instanciar o modelo

xgb = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42, learning_rate=0.05)

xgb.fit(train[feats], train['BAD'])

scores = cross_val_score(xgb, train[feats], train['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()
# Fazendo predições

preds = xgb.predict(test[feats])

preds
# Verificando o real

test['BAD'].head(3)
# Medir o desempenho do modelo XGB

accuracy_score(test['BAD'], preds)
# Instanciar o modelo

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
# Treinando o modelo

rf.fit(train[feats], train['BAD'])
# Fazendo previsões em cima dos dados de validação

predstree = rf.predict(test[feats])
# Verificando as previsoes

predstree

# Verificando o real

test['BAD'].head(3)
# Aplicando a metrica

mean_squared_error(test['BAD'], preds)**(1/2)