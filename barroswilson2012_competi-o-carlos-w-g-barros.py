# Importação de Pacotes



import pandas as pd

import pandas_profiling as ppf

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import scikitplot as splt



%matplotlib inline



import os

print(os.listdir("../input"))



# Criando os dataframes 



df_sub=pd.read_csv('../input/sample_submission.csv', thousands=',')

df_train = pd.read_csv('../input/train.csv', thousands=',')

df_test = pd.read_csv('../input/test.csv', thousands=',')
# bases de treino, base de teste sem o campo nota de matemática e vetor de relacionamento da base de matemática de teste

# Processo treina o modelo, roda o modelos com os dados de teste e depois comparar com os dados reais da base de teste que encontran-se no vetor



df_train.shape,df_test.shape,df_sub.shape
df_train.info()
# Valores missing na tabela de teino

df_train.isnull().sum().to_frame('Qtd. Missing').T
df_test.info()
# Valores missing na tabela de teste

df_test.isnull().sum().to_frame('Qtd. Missing').T
# vetor com os dados observados da nota de matemática referentes a tabela de teste

df_sub.info()
df = df_train.append(df_test, sort=True)

df.shape
# outra forma seria usar os comandos abaixo no entanto deveriamos criar um campo matemática na tabela de teste para que funcione

# df_base = pd.concat([df_train,df_test])
df.sample(5)
#verificando visualmente os missing campo a campo

import missingno as msno



msno.matrix(df,figsize=(12,5))
df.info()
# Verificando formato de campos

print('*** Frequencia ***')

print(df.dtypes.value_counts())



print('%%%%%%%%%%%%%%%%%%%')

print(df.dtypes.value_counts(normalize=True).apply("{:.2%}".format))

df.columns
# tratamento da base

df['populacao'] = df['populacao'].str.replace(',','').str.replace('.','').apply(lambda x: x.split('(')[0]).astype(int)

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('#DIV/0!','').str.rstrip('%')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].convert_objects(convert_numeric=True)/100

df['exp_vida']=df['exp_vida'].round(2)

df['codigo_mun'] = df['codigo_mun'].astype(object)

df['ct_porte'] = df['porte'].astype('category').cat.codes

df['ct_regiao'] = df['regiao'].astype('category').cat.codes

df['ct_estado'] = df['estado'].astype('category').cat.codes



df.info()
#verificando visualmente os missing campo a campo

msno.matrix(df,figsize=(12,5))
# Regra para substituir os valores nulos pela média (não incluindo nota_mat)

preencher=['comissionados_por_servidor', 'densidade_dem', 'exp_anos_estudo',

           'exp_vida', 'gasto_pc_educacao', 'gasto_pc_saude', 'hab_p_medico',

           'participacao_transf_receita', 'perc_pop_econ_ativa','servidores']



for c in preencher:

    df[c] = df.groupby(['estado', 'porte'])[c].transform(lambda x:x.fillna(x.mean()))
df.info()
#verificando visualmente os missing campo a campo

msno.matrix(df,figsize=(12,5))
df.info()
train = df[~df['nota_mat'].isnull()]

train = train.fillna(-1)

test = df[df['nota_mat'].isnull()]

test = test.fillna(-1)

from sklearn.model_selection import train_test_split

treino, valid = train_test_split(train, random_state=42)

treino.shape, valid.shape, test.shape
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=25, n_jobs=-1, n_estimators=300)
removed_cols = ['regiao','estado','municipio','porte','codigo_mun','densidade_dem',

                'comissionados_por_servidor','nota_mat']

cols = []

for c in treino.columns:

    if c not in removed_cols:

        cols.append(c)

cols



feats = [c for c in treino.columns if c not in removed_cols]
feats
rfc.fit(treino[feats], treino['nota_mat'])
preds = rfc.predict(valid[feats])
from sklearn.metrics import accuracy_score

accuracy_score(valid['nota_mat'], preds)
test['nota_mat'] = rfc.predict(test[feats])
test.shape
test[['codigo_mun','nota_mat']].to_csv('modelo.csv', index=False)