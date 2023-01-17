# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Dados
treino = pd.read_csv('../input/train.csv')
teste = pd.read_csv('../input/test.csv')
df = treino.append(teste)
df.shape
df.head().T
#Removendo primeira coluna
df = df.drop('Unnamed: 0', axis = 1)
df.info()
df['populacao'] = df['populacao'].astype('float64')
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))
df['codigo_mun'] = df['codigo_mun'].values.astype('int64')
df['area'] = df['area'].apply(lambda x: x.replace(',', ''))
df['area'] = df['area'].values.astype('float64')
# Quantidade de missing
df.isnull().sum()


#Tipos das variáveis
df.info()
# Medidas descritivas
df.describe().T
#Matriz de correlação
df.corr()
#Nota mat tem alta correlação com: 
#  IDHM: 0.773060
#  anos_estudo_empreendedor: 0.706861
#  perc_pop_econ_ativa: 0.706240
#  exp_vida: 0.703872
#  indice_governanca: 0.676149
plt.scatter('nota_mat', 'idhm', data = df)
plt.scatter('nota_mat', 'anos_estudo_empreendedor', data = df)
plt.scatter('nota_mat', 'perc_pop_econ_ativa', data = df)
plt.scatter('nota_mat', 'exp_vida', data = df)
plt.scatter('nota_mat', 'indice_governanca', data = df)
sns.boxplot(y='nota_mat', x='porte', data=df)
sns.boxplot(y='nota_mat', x='capital', data=df)
sns.boxplot(y='nota_mat', x='estado', data=df)
sns.boxplot(y='nota_mat', x='regiao', data=df)
# Remover linhas com missing
df_a = df.copy()
# Codificando texto em número
for col in df_a.columns:
    if df_a[col].dtype == 'object':
        df_a[col] = df_a[col].astype('category').cat.codes
df_a.info()
df_a.dropna(how='any', inplace=True)
df_a.shape # Perde muitos dados. Não é muito bom.
# Preenchendo os dados faltantes com -1
df_b = df.copy()
# Codificando texto em número
for col in df_b.columns:
    if df_b[col].dtype == 'object':
        df_b[col] = df_b[col].astype('category').cat.codes
df_b.info()
# Substituir dado faltante por -10000
df_b.fillna(-10000, inplace=True)
# Separando novamente as bases
test = df_b[df_b['nota_mat'] == -10000]
test.shape
test.head()
tr = df_b[df_b['nota_mat'] != -10000]
tr.shape
tr.head()
# Separando dados de treino e validação
train2, valid2 = train_test_split(tr, random_state=42)
# Instanciando a RandomForest
rf2 = RandomForestRegressor(random_state = 54, n_jobs = -1, max_features = 18, max_depth = 20, n_estimators = 100)
# Separando as features
feats2 = [c for c in tr.columns if c not in ['nota_mat', 'municipio', 'codigo_mun', 
                                               'densidade_dem', 'comissionados_por_servidor']]
# Treinando o modelo
rf2.fit(train2[feats2], train2['nota_mat'])
mean_squared_error(rf2.predict(valid2[feats2]), valid2['nota_mat'])**(1/2)
pd.Series(rf2.feature_importances_, index=feats2).sort_values().plot.barh()
#Substituir os dados faltantes pela média ou mediana (numéricas) 
sns.boxplot(y='participacao_transf_receita', x='porte', data=df)
df.groupby('porte').participacao_transf_receita.mean()
df.groupby('porte').participacao_transf_receita.median()
sns.boxplot(y='servidores', x='porte', data=df)
df.groupby('porte').servidores.median()
df.groupby('porte').servidores.mean()
sns.boxplot(y='gasto_pc_saude', x='regiao', data=df)
df.groupby('regiao').gasto_pc_saude.mean()
df.groupby('regiao').gasto_pc_saude.median()
sns.boxplot(y='hab_p_medico', x='regiao', data=df)
df.groupby('regiao').hab_p_medico.mean()
df.groupby('regiao').hab_p_medico.median()
sns.boxplot(y='gasto_pc_educacao', x='regiao', data=df)
df.groupby('regiao').gasto_pc_educacao.mean()
df.groupby('regiao').gasto_pc_educacao.median()
sns.boxplot(y='idhm', x='regiao', data=df)
df.groupby('regiao').idhm.mean()
# Separando base por porte para imputação
df_c = df.copy()
pequeno1 = df_c[df_c['porte'] == 'Pequeno porte 1']
pequeno2 = df_c[df_c['porte'] == 'Pequeno porte 2']
medio = df_c[df_c['porte'] == 'Médio porte']
grande = df_c[df_c['porte'] == 'Grande porte']
#Preenchendo os valores
pequeno1['participacao_transf_receita'].fillna(pequeno1['participacao_transf_receita'].mean(), inplace=True)
pequeno2['participacao_transf_receita'].fillna(pequeno2['participacao_transf_receita'].mean(), inplace=True)
medio['participacao_transf_receita'].fillna(medio['participacao_transf_receita'].mean(), inplace=True)
grande['participacao_transf_receita'].fillna(grande['participacao_transf_receita'].mean(), inplace=True)

pequeno1['servidores'].fillna(pequeno1['servidores'].median(), inplace=True)
pequeno2['servidores'].fillna(pequeno2['servidores'].median(), inplace=True)
medio['servidores'].fillna(medio['servidores'].median(), inplace=True)
grande['servidores'].fillna(grande['servidores'].median(), inplace=True)

#Juntando as bases
df_c = pd.concat([pequeno1, pequeno2, medio, grande])
df_c.isnull().sum()
# Agora, separando por região
co = df_c[df_c['regiao'] == 'CENTRO-OESTE']
nord = df_c[df_c['regiao'] == 'NORDESTE']
norte = df_c[df_c['regiao'] == 'NORTE']
sud = df_c[df_c['regiao'] == 'SUDESTE']
sul = df_c[df_c['regiao'] == 'SUL']
#Preenchendo os valores
co['gasto_pc_saude'].fillna(co['gasto_pc_saude'].mean(), inplace=True)
nord['gasto_pc_saude'].fillna(nord['gasto_pc_saude'].mean(), inplace=True)
norte['gasto_pc_saude'].fillna(norte['gasto_pc_saude'].mean(), inplace=True)
sud['gasto_pc_saude'].fillna(sud['gasto_pc_saude'].mean(), inplace=True)
sul['gasto_pc_saude'].fillna(sul['gasto_pc_saude'].mean(), inplace=True)

co['hab_p_medico'].fillna(co['hab_p_medico'].median(), inplace=True)
nord['hab_p_medico'].fillna(nord['hab_p_medico'].median(), inplace=True)
norte['hab_p_medico'].fillna(norte['hab_p_medico'].median(), inplace=True)
sud['hab_p_medico'].fillna(sud['hab_p_medico'].median(), inplace=True)
sul['hab_p_medico'].fillna(sul['hab_p_medico'].median(), inplace=True)

co['gasto_pc_educacao'].fillna(co['gasto_pc_educacao'].mean(), inplace=True)
nord['gasto_pc_educacao'].fillna(nord['gasto_pc_educacao'].mean(), inplace=True)
norte['gasto_pc_educacao'].fillna(norte['gasto_pc_educacao'].mean(), inplace=True)
sud['gasto_pc_educacao'].fillna(sud['gasto_pc_educacao'].mean(), inplace=True)
sul['gasto_pc_educacao'].fillna(sul['gasto_pc_educacao'].mean(), inplace=True)

co['idhm'].fillna(co['idhm'].mean(), inplace=True)
nord['idhm'].fillna(nord['idhm'].mean(), inplace=True)
norte['idhm'].fillna(norte['idhm'].mean(), inplace=True)
sud['idhm'].fillna(sud['idhm'].mean(), inplace=True)
sul['idhm'].fillna(sul['idhm'].mean(), inplace=True)

#Juntando as bases
df_c = pd.concat([co, nord, norte, sud, sul])
df_c.isnull().sum()
df_c.info()
#Preenchendo os valores das demais com a média
df_c['perc_pop_econ_ativa'].fillna(df_c['perc_pop_econ_ativa'].mean(), inplace=True)
df_c['exp_vida'].fillna(df_c['exp_vida'].mean(), inplace=True)
df_c['exp_anos_estudo'].fillna(df_c['exp_anos_estudo'].mean(), inplace=True)
# Preenchendo as demais com -1
df_c.fillna(-1, inplace=True)
for c in df_c.columns:
    if df_c[c].dtypes == 'object':
        df_c[c] = df_c[c].astype('category').cat.codes
# Separando novamente as bases
test_c = df_c[df_c['nota_mat'] == -1]
test_c.shape
train_c = df_c[df_c['nota_mat'] != -1]
train_c.shape
train_c['nota_mat'] = np.log(train_c['nota_mat'])
# Separando dados de treino e validação
train3, valid3 = train_test_split(train_c, random_state=42)
# Instanciando a RandomForest
rf3 = RandomForestRegressor(random_state = 54, n_jobs = -1, max_features = 18, max_depth = 20, 
                            n_estimators = 100)
# Separando as features
feats3 = [c for c in train_c.columns if c not in ['nota_mat', 'municipio', 'codigo_mun',  
                                              'densidade_dem', 'comissionados_por_servidor', 'pib']]
# Treinando o modelo
rf3.fit(train3[feats3], train3['nota_mat'])
mean_squared_error(rf3.predict(valid3[feats3]), valid3['nota_mat'])**(1/2)
pd.Series(rf3.feature_importances_, index=feats3).sort_values().plot.barh()
test_c['nota_mat'] = np.exp(rf3.predict(test_c[feats3]))
test_c.head()
test_c[['codigo_mun', 'nota_mat']].to_csv('Carolina_ASilva4.csv', index = False)
