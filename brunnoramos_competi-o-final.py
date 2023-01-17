# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# A base de treino será chamada de df

df = pd.read_csv('../input/train.csv')

df.head()
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
df = pd.concat([test,train])
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))
df['codigo_mun'] = df['codigo_mun'].values.astype('int64')
df['area'] = df['area'].apply(lambda x: x.replace(',', ''))
df['area'] = df['area'].values.astype('float64')
#retorna (n row, n col)

df.shape
#Retorna informações do dataset

df.info()
#Retorna estatísticas descritivas para as variáveis quantitativas

df.describe()
# Retorna a soma dos valores nulos em cada coluna

df.isnull().sum()



# As seguintes variáveis possuem valores nulos

# densidade_dem, participacao_transf_receita, servidores, perc_pop_econ_ativa gasto_pc_saude, 

# hab_p_medico,exp_vida, gasto_pc_educacao, exp_anos_estudo  

#Retorna o tipo de variável de cada coluna

df.dtypes
# Média da participação transferencia receita por porte

df.groupby('porte').participacao_transf_receita.mean()  
# Média da participação transferencia receita por regiao

df.groupby('regiao').participacao_transf_receita.mean()  
sns.boxplot(y='participacao_transf_receita', x = 'porte', data=df)
## Quanto maior o porte, menor a participação_transf_receita
# Mostra a distribuição de municipios por porte

df['porte'].value_counts()
# Média da participação transferencia receita por porte

sns.boxplot(y='servidores', x = 'porte', data=df)
df.groupby('porte').servidores.median()  
df.groupby('porte').servidores.mean()  
df.groupby('porte').gasto_pc_saude.mean()    
df.groupby('regiao').gasto_pc_saude.mean()    
sns.boxplot(y='gasto_pc_saude', x = 'regiao', data=df)
sns.boxplot(y='gasto_pc_saude', x = 'porte', data=df)
df.groupby('porte').hab_p_medico.mean()   
df.groupby('regiao').hab_p_medico.mean()   
df.groupby('capital').hab_p_medico.mean()   
sns.boxplot(y='hab_p_medico', x = 'porte', data=df)
sns.boxplot(y='hab_p_medico', x = 'regiao', data=df)
df.groupby('porte').gasto_pc_educacao.mean()    
df.groupby('regiao').gasto_pc_educacao.mean()    
df.groupby('porte').gasto_pc_educacao.median()    
df.groupby('regiao').gasto_pc_educacao.median()   
sns.boxplot(y='gasto_pc_educacao', x = 'regiao', data=df)
sns.boxplot(y='gasto_pc_educacao', x = 'porte', data=df)
df.groupby('porte').idhm.mean()    
df.groupby('regiao').idhm.mean()  
sns.boxplot(y='idhm', x = 'regiao', data=df)
df.corr()
plt.scatter('nota_mat', 'idhm', data = df)
plt.scatter('nota_mat', 'anos_estudo_empreendedor', data = df)
plt.scatter('nota_mat', 'perc_pop_econ_ativa', data = df)
plt.scatter('nota_mat', 'exp_vida', data = df)
# Categorizar a variável 'Regiao'

df['regiao'].head()

df['regiao'].astype('category').cat.codes.value_counts()
#Criando uma função para transformar todas as colunas que tem tipo objeto

def transf_num(df):

    for i in df.columns:

        if df[i].dtypes == 'object':

         df[i] = df[i].astype('category').cat.codes
transf_num(df)

df.dtypes
## Criando cópias dos bancos de dados para realizar as diferentes abordagens ##

df1 = df.copy()

df2 = df.copy()

df3 = df.copy()
## sempre que convertemos uma coluna de letras para números ele muitas vezes assume o valor -1

## utilizaremos a transformação em -2 para não confundir as informações

df1.min().min()
#substituindo os valores nulos por -2 saberemos exatamente quais os valores eram nulos,

#e os valores 'nulos' estarão fora do range do modelo que iremos construir

df1.fillna(-2, inplace=True)

df1.head()

df1.isnull().sum()
df1.corr()
#Criar histograma da variável nota_mat

df1['nota_mat'].hist()
df1['nota_mat'].plot.box()
import matplotlib.pyplot as plt

import seaborn as sns
removed_cols = ['Unnamed: 0', 'codigo_mun', 'nota_ciencias', 'nota_humanas', 'nota_linguagem',

               'nota_mat', 'nota_redacao']

feats1 = [c for c in df1.columns if c not in removed_cols]
from sklearn.model_selection import train_test_split
train1, valid1 = train_test_split(df1, random_state=42)
train1.shape, valid1.shape
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(train1[feats1], train1['nota_mat'])
from sklearn.metrics import mean_squared_error
mean_squared_error(rf.predict(valid1[feats1]), valid1['nota_mat'])**(1/2)
rf = RandomForestRegressor(random_state=42, n_estimators=100, max_features=0.8)
#padrão do numero de árvores

#ao alterar a semente é normal o erro mudar pq muda a amostra. Mas ganhos pequenos são aleatórios 

rf.fit(train1[feats1], train1['nota_mat'])

mean_squared_error(rf.predict(valid1[feats1]),valid1['nota_mat'])**(1/2)
#Verifica quais os critérios mais importãntes utlizados pelo modelo para a tomada de decisão

pd.Series(rf.feature_importances_, index=feats1).sort_values().plot.barh()
df2.fillna(-1000, inplace=True)

df2.head()

df2.isnull().sum()
removed_cols = ['Unnamed: 0', 'codigo_mun', 'nota_ciencias', 'nota_humanas', 'nota_linguagem',

               'nota_mat', 'nota_redacao']

feats2 = [c for c in df2.columns if c not in removed_cols]
train2, valid2 = train_test_split(df2, random_state=42)
train1.shape, valid1.shape
rf.fit(train2[feats2], train2['nota_mat'])
mean_squared_error(rf.predict(valid2[feats2]), valid2['nota_mat'])**(1/2)
rf = RandomForestRegressor(random_state=42, n_jobs= -1, max_features = 18, max_depth = 20, n_estimators = 100)
#padrão do numero de árvores

#ao alterar a semente é normal o erro mudar pq muda a amostra. Mas ganhos pequenos são aleatórios 

rf.fit(train2[feats2], train2['nota_mat'])

mean_squared_error(rf.predict(valid2[feats2]),valid2['nota_mat'])**(1/2)
#Verifica quais os critérios mais importãntes utlizados pelo modelo para a tomada de decisão

pd.Series(rf.feature_importances_, index=feats1).sort_values().plot.barh()
## Rodando novamente estes comandos apenas por conta da categorização

df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

df = pd.concat([test,train])

df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))

df['codigo_mun'] = df['codigo_mun'].values.astype('int64')

df['area'] = df['area'].apply(lambda x: x.replace(',', ''))

df['area'] = df['area'].values.astype('float64')

df3 = df.copy()
# A variável ranking IGM e indice_governanca oculta os valores outliers utilizando valores nulos, dessa forma utilizaremos um valor

# extremo para preencher seus valores nulos

df3['ranking_igm'].fillna(-1000, inplace=True)

df3['indice_governanca'].fillna(-1000, inplace=True)

df3.isnull().sum()
## A variável servidores possui 131 valores nulos, estes iremos tratar utilizando a média de servidores

## baseada no porte do município

# Mostra a distribuição de municipios por porte

df3['porte'].value_counts()
# Média de servidores por porte de municipio

df3.groupby('porte').servidores.mean()        
df3.groupby('porte').servidores.mean().plot.bar()
## Segmentando df3 nos 4 tipos de porte para realizar a imputação

pequeno_1 = df3[df3['porte']== 'Pequeno porte 1']

pequeno_2 = df3[df3['porte']== 'Pequeno porte 2']

medio = df3[df3['porte']== 'Médio porte']

grande = df3[df3['porte']== 'Grande porte']
## Preenchendo os valores nulos da variável servidores com a mediana de cada porte

pequeno_1['servidores'].fillna(pequeno_1['servidores'].median(), inplace=True)

pequeno_1['participacao_transf_receita'].fillna(pequeno_1['participacao_transf_receita'].mean(), inplace=True)
grande['servidores'].median()
pequeno_2['servidores'].fillna(pequeno_2['servidores'].median(), inplace=True)

pequeno_2['participacao_transf_receita'].fillna(pequeno_2['participacao_transf_receita'].mean(), inplace=True)
medio['servidores'].fillna(medio['servidores'].median(), inplace=True)

medio['participacao_transf_receita'].fillna(medio['participacao_transf_receita'].mean(), inplace=True)
grande['servidores'].fillna(grande['servidores'].median(), inplace=True)

grande['participacao_transf_receita'].fillna(grande['participacao_transf_receita'].mean(), inplace=True)
## juntando novamente as 4 bases no df3

df3 = pd.concat([pequeno_1,pequeno_2,medio, grande])

df3.isnull().sum()
## Segmentando df3 nos 5 tipos de região para realizar a imputação

CENTRO_OESTE = df3[df3['regiao']== 'CENTRO-OESTE']

NORDESTE = df3[df3['regiao']== 'NORDESTE']

NORTE = df3[df3['regiao']== 'NORTE']

SUDESTE = df3[df3['regiao']== 'SUDESTE']

SUL = df3[df3['regiao']== 'SUL']
## Preenchendo os valores nulos da variável servidores com a mediana de cada porte

CENTRO_OESTE['gasto_pc_educacao'].fillna(CENTRO_OESTE['gasto_pc_educacao'].mean(), inplace=True)

CENTRO_OESTE['gasto_pc_saude'].fillna(CENTRO_OESTE['gasto_pc_saude'].mean(), inplace=True)

CENTRO_OESTE['idhm'].fillna(CENTRO_OESTE['idhm'].mean(), inplace=True)

CENTRO_OESTE['hab_p_medico'].fillna(CENTRO_OESTE['hab_p_medico'].mean(), inplace=True)

## Preenchendo os valores nulos da variável servidores com a mediana de cada porte

NORDESTE['gasto_pc_educacao'].fillna(NORDESTE['gasto_pc_educacao'].mean(), inplace=True)

NORDESTE['gasto_pc_saude'].fillna(NORDESTE['gasto_pc_saude'].mean(), inplace=True)

NORDESTE['idhm'].fillna(NORDESTE['idhm'].mean(), inplace=True)

NORDESTE['hab_p_medico'].fillna(NORDESTE['hab_p_medico'].mean(), inplace=True)
NORTE['gasto_pc_educacao'].fillna(NORTE['gasto_pc_educacao'].mean(), inplace=True)

NORTE['gasto_pc_saude'].fillna(NORTE['gasto_pc_saude'].mean(), inplace=True)

NORTE['idhm'].fillna(NORTE['idhm'].mean(), inplace=True)

NORTE['hab_p_medico'].fillna(NORTE['hab_p_medico'].mean(), inplace=True)
SUDESTE['gasto_pc_educacao'].fillna(SUDESTE['gasto_pc_educacao'].mean(), inplace=True)

SUDESTE['gasto_pc_saude'].fillna(SUDESTE['gasto_pc_saude'].mean(), inplace=True)

SUDESTE['idhm'].fillna(SUDESTE['idhm'].mean(), inplace=True)

SUDESTE['hab_p_medico'].fillna(SUDESTE['hab_p_medico'].mean(), inplace=True)
SUL['gasto_pc_educacao'].fillna(SUL['gasto_pc_educacao'].mean(), inplace=True)

SUL['gasto_pc_saude'].fillna(SUL['gasto_pc_saude'].mean(), inplace=True)

SUL['idhm'].fillna(SUL['idhm'].mean(), inplace=True)

SUL['hab_p_medico'].fillna(SUL['hab_p_medico'].mean(), inplace=True)
## juntando novamente as 4 bases no df3

df3 = pd.concat([CENTRO_OESTE,NORDESTE,NORTE, SUDESTE, SUL])

df3.isnull().sum()
# Categorizar a variável 'Regiao'

df3['regiao'].head()

df3['regiao'].astype('category').cat.codes.value_counts()
#Criando uma função para transformar todas as colunas que tem tipo objeto

def transf_num(df3):

    for i in df3.columns:

        if df3[i].dtypes == 'object':

         df3[i] = df3[i].astype('category').cat.codes
transf_num(df3)

df3.dtypes
#Preenchendo os valores nulos restantes com -suas respectivas médias

df3['densidade_dem'].fillna(df3['densidade_dem'].mean(), inplace=True)

df3['exp_vida'].fillna(df3['exp_vida'].mean(), inplace=True)

df3['exp_anos_estudo'].fillna(df3['exp_anos_estudo'].mean(), inplace=True)

df3['perc_pop_econ_ativa'].fillna(df3['perc_pop_econ_ativa'].mean(), inplace=True)

df3.head()

df3.isnull().sum()
#substituindo os valores nulos por -2 saberemos exatamente quais os valores eram nulos,

#e os valores 'nulos' estarão fora do range do modelo que iremos construir

df3.fillna(-2, inplace=True)

df3.head()

df3.isnull().sum()
test = df3[df3['nota_mat']== -2]

df3 = df3[df3['nota_mat']!= -2]
test.head()

test.shape
df3.head()
df3.shape
removed_cols = ['Unnamed: 0', 'codigo_mun', 'nota_ciencias', 'nota_humanas', 'nota_linguagem',

               'nota_mat', 'nota_redacao']

feats3 = [c for c in df3.columns if c not in removed_cols]
from sklearn.model_selection import train_test_split

train3, valid3 = train_test_split(df3, random_state=42)
train3.shape, valid3.shape
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_jobs= -1, max_features = 18, max_depth = 20, n_estimators = 100)

rf.fit(train3[feats3], train3['nota_mat'])
from sklearn.metrics import mean_squared_error

mean_squared_error(rf.predict(valid3[feats3]), valid3['nota_mat'])**(1/2)
#padrão do numero de árvores

#ao alterar a semente é normal o erro mudar pq muda a amostra. Mas ganhos pequenos são aleatórios 

rf.fit(train3[feats3], train3['nota_mat'])

mean_squared_error(rf.predict(valid3[feats3]),valid3['nota_mat'])**(1/2)
#Verifica quais os critérios mais importãntes utlizados pelo modelo para a tomada de decisão

pd.Series(rf.feature_importances_, index=feats3).sort_values().plot.barh()
test['nota_mat'] = rf.predict(test[feats3])
test[['codigo_mun','nota_mat']].to_csv('kalyxton.csv', index=False)