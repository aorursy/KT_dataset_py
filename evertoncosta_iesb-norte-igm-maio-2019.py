# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy  as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df    = pd.read_csv('../input/train.csv', parse_dates=[0]) #lê coluna em formato de data

teste = pd.read_csv('../input/test.csv', parse_dates=[0])
df1 = df.append(teste, sort=False)
df1.shape
df1.head().T
df1.tail().T
df1.sample(20).T
df1.info()
df1.iloc[:10, -2:].T
df1.iloc[-10:, -2:].T
# Analisando a coluna regiao

df1['regiao'].value_counts()
# Analisando a coluna estado

df1['estado'].value_counts()
# Analisando a coluna codigo_mun

df1['codigo_mun'].value_counts()
df1['populacao'].sample(10)
df1['populacao'] = df1.populacao.astype(int)
df1['populacao'].head()
df1['populacao'].sample(10)
df1['populacao'] = df1['populacao'].str.replace(',','')

df1['populacao'] = df1.populacao.astype(int)
# df1['populacao'] = df1['populacao'].str.split('(').str[0].astype(int)

df1['populacao'] = df1['populacao'].str.split('(').str[0]

df1['populacao'] = df1.populacao.astype(int)
df1['populacao'] = df1['populacao'].str.replace('.','')

df1['populacao'] = df1.populacao.astype(int)
df1['populacao'].sample(10)
df1['area'] = df1['area'].str.replace(',','')

df1['area'] = df1.area.astype(float)
df1['densidade_dem'] = df1['densidade_dem'].str.replace(',','')

df1['densidade_dem'] = df1.densidade_dem.astype(float)
df1['servidores_na'] = np.where(df1['servidores'].isna(), -1, 0)

df1['servidores']    = np.where(df1['servidores'].isna(), 

                                round(df1['populacao'] * df1.servidores.mean() / df1.populacao.mean(), 0),

                                df1['servidores'])
df1['gasto_pc_educacao_na'] = np.where(df1['gasto_pc_educacao'].isna(), -1, 0)

df1['gasto_pc_educacao']    = np.where(df1['gasto_pc_educacao'].isna(), 

                                       round(df1['populacao'] * df1.gasto_pc_educacao.mean() / df1.populacao.mean(), 2),

                                       df1['gasto_pc_educacao'])
df1['densidade_dem_na'] = np.where(df1['densidade_dem'].isna(), -1, 0)

df1['densidade_dem']    = np.where(df1['densidade_dem'].isna(), 

                                   round(df1['populacao'] / df1['area'], 2),

                                   df1['densidade_dem'])
df1['hab_p_medico_na'] = np.where(df1['hab_p_medico'].isna(), -1, 0)

df1['hab_p_medico']    = np.where(df1['hab_p_medico'].isna(), 

                                 round(df1['populacao'] * df1.hab_p_medico.mean() / df1.populacao.mean(), 0),

                                 df1['hab_p_medico'])
df1['perc_pop_econ_ativa'] = np.where(df1['perc_pop_econ_ativa'].isna(), 

                                      -1, 

                                      df1['perc_pop_econ_ativa'])
df1['participacao_transf_receita'] = np.where(df1['participacao_transf_receita'].isna(), 

                                              -1, 

                                              df1['participacao_transf_receita'])
df1['gasto_pc_saude'] = np.where(df1['gasto_pc_saude'].isna(), 

                                 -1, 

                                 df1['gasto_pc_saude'])
df1['exp_vida'] = np.where(df1['exp_vida'].isna(), 

                           df1['exp_vida'].mean(), 

                           df1['exp_vida'])
df1['exp_anos_estudo'] = np.where(df1['exp_anos_estudo'].isna(), 

                                  df1['exp_anos_estudo'].mean(), 

                                  df1['exp_anos_estudo'])
df1['comissionados_por_servidor'] = df1['comissionados'] / df1['servidores']
df1.info()
for col in df1.columns:

    if df1[col].dtype == 'object':

        df1[col] = df1[col].astype('category').cat.codes
test1 = df1[df1['nota_mat'].isnull()]

df2   = df1[~df1['nota_mat'].isnull()]
# Separando o dataframe

from sklearn.model_selection import train_test_split

train, test = train_test_split(df2, test_size=0.20, random_state = 42)
# Separando os dados para a validação

train, valid = train_test_split(train, test_size=0.20, random_state = 42)
train.shape, test.shape, valid.shape, test1.shape
removed_cols = ['nota_mat', 'nota_mat']
# Lista das colunas a serem usadas para o treino

feats = [c for c in df1.columns 

             if c not in removed_cols]
removed_cols
feats
# Importando o RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
# Instanciando um objeto RandomForest

rf = RandomForestClassifier(n_estimators=150, min_samples_split=3, max_depth=4, random_state=42)
# Treinando o modelo

rf.fit(train[feats], train['nota_mat'])
preds = rf.predict(valid[feats])
# Avaliando o desempenho do modelo

from sklearn.metrics import accuracy_score
# Avaliando

accuracy_score(valid['nota_mat'], preds)
#Avaliando o modelo com relação aos dados de teste

accuracy_score(test['nota_mat'], rf.predict(test[feats]))
# Avaliar o modelo com realção ao classificador da maioria

test['nota_mat'].value_counts()
(test['nota_mat'] == 0).mean()
import matplotlib.pyplot as plt
# Avaliando a importancia das variaveis

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
preds_trab = rf.predict(test1[feats])
test1['nota_mat'] = preds_trab
test1.shape
test1[['codigo_mun', 'nota_mat']].to_csv('trabalho1.csv', index=False)