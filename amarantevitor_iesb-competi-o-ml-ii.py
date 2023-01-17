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
# Carregar os dados

df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df.shape, test.shape
df.info()
df.head().T
# Pré-processamento dos dados



# Juntar os datraframes

df = df.append(test)
df.head().T
df.tail().T
df.info()
# Convertendo area para float

df['area'] = df['area'].str.replace(',','')

df['densidade_dem'] = df['densidade_dem'].str.replace(',','')

df['area'] = df['area'].astype(float)

df['densidade_dem'] = df['densidade_dem'].astype(float)
df['populacao'] = df['populacao'].str.replace(',','')

df['populacao'] = df['populacao'].str.replace('(2)','')

df['populacao'] = df['populacao'].str.replace('(','')

df['populacao'] = df['populacao'].str.replace(')','')

df['populacao'] = df['populacao'].str.replace('.','')

df['populacao'] = df['populacao'].astype(int)
df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('#DIV/','')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('!','')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('%','')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].astype(float)

df['comissionados_por_servidor'] = df['comissionados_por_servidor']/100
# Transformar as variáveis porte e regiao em dummy

df_porte = pd.get_dummies(df['porte'])

df_regiao = pd.get_dummies(df['regiao'])
# Concatenar os dataframes

df = pd.concat([df, df_porte], axis=1)

df = pd.concat([df, df_regiao], axis=1)
#Correlação entre variaveis

df.corr()
# Importando as bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns
# Plotar as correlaçoes

f,ax = plt.subplots(figsize=(25,13))

sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=.7)
df.head()
df.info()
# Separar os dataframes pelo valores nulos

test = df[df['nota_mat'].isnull()]

df = df[~df['nota_mat'].isnull()]
df.shape, test.shape
# Separando o df em treino e validação

from sklearn.model_selection import train_test_split

train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape
# Selecionar as colunas a serem usadas no treinamento e validação

# Lista das colunas não usadas

removed_cols = ['municipio', 'estado', 'codigo_mun', 'porte', 'regiao', 'nota_mat','capital']



#Lista das features

feats = [c for c in df.columns if c not in removed_cols]
# Importando os modelos

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier 
# Dicionario de modelos

models = {'RandomForest': RandomForestClassifier (random_state=42, min_samples_leaf=8,

                                                  min_samples_split=3, n_jobs=-1, n_estimators=75),

          'ExtraTrees': ExtraTreesClassifier (random_state=42, n_jobs=-1),

          'GradientBoosting': GradientBoostingClassifier (random_state=42),

          'DecisionTree': DecisionTreeClassifier (random_state=42),

          'AdaBoost': AdaBoostClassifier (random_state=42),

          'KNM 11': KNeighborsClassifier (n_neighbors=11, n_jobs=-1),}
# Impotando metrica

# Importando Métrica de Acurácia

from sklearn.metrics import accuracy_score
# Função para treino dos modelos

def run_model(models, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return accuracy_score(valid[y_name], preds)
# Executando os modelos

scores = []

for name, model in models.items():

    score = run_model(model, train.fillna(-1), valid.fillna(-1), feats, 'nota_mat')

    scores.append(score)

    print(name, ':', score) 
# Juntando a tabela treino com validação

train = train.append(valid)
# Parametrizando a RandomForestClassifier que foi o modelo com o melhor Score

rf = RandomForestClassifier (random_state=42, min_samples_leaf=8, min_samples_split=3, n_jobs=-1, n_estimators=200)
# Treinando o modelo

rf.fit(train[feats].fillna(-1), train['nota_mat'])
# Aplicando o modelo na base de teste

test['nota_mat'] = rf.predict(test[feats].fillna(-1))
# Gerando o arquivo csv para submeter a competição

test[['codigo_mun','nota_mat']].to_csv('randomforest.csv', index=False)