# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importando os arquivos



# Dados de treino

df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')



# Dados de teste

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')



df.shape, test.shape
# Verificando quantidades e tipos da base de treino

df.info()
# Verificando quantidades e tipos da base de teste

test.info()
# Verificando os dados na base de treino

df.head()
# Verificando os dados na base de teste

test.head()
# Vamos olhar a variável count

df['count'].plot.hist(bins=50)
# Qual os horários tiveram maiores aluguies de bicicleta

df.nlargest(5, 'count')
# Aplicando a escala logaritmica nos valores de count

df['count'] = np.log(df['count'])
# Vamos olhar a variável count

df['count'].plot.hist(bins=50)
# Juntar os dataframes para realizar transformações nas variáveis de entrada

df_total = df.append(test)
# Converter a coluna datetime

df_total['datetime'] = pd.to_datetime(df_total['datetime'])
# Verificando a conversão

df_total.info()
# Feature Engineering



# Criar novas colunas com base na coluna datetime

df_total['year'] = df_total['datetime'].dt.year

df_total['month'] = df_total['datetime'].dt.month

df_total['day'] = df_total['datetime'].dt.day

df_total['dayofweek'] = df_total['datetime'].dt.dayofweek

df_total['hour'] = df_total['datetime'].dt.hour



# Verificando as colunas criadas

df_total.info()
# Separando os dataframes



# Dataframe de teste

test = df_total[df_total['count'].isnull()]



# Dataframe de treino

df = df_total[~df_total['count'].isnull()]



df.shape, test.shape
# Dividindo o dataframe de treino em treino e validação



# Vamos usar a função train_test_split

from sklearn.model_selection import train_test_split



# Dividir a base

train, valid = train_test_split(df, random_state=42)



# Verificando os dataframes

train.shape, valid.shape
# Selecionando as colunas para treinamento

removed = ['casual', 'registered', 'count', 'datetime']



# Lista de colunas para treino

feats = [col for col in train.columns if col not in removed]
# Usando o modelo Random Forest



# Importar o modelo

from sklearn.ensemble import RandomForestRegressor



# Instanciar o modelo

rf = RandomForestRegressor(random_state=42)
# Treinando o modelo

rf.fit(train[feats], train['count'])
# Fazendo previsões de acordo com o modelo treinado

preds = rf.predict(valid[feats])
# Olhando as previsões

preds
# Verificando o desempenho



# Importando a métrica

from sklearn.metrics import mean_squared_error



# Aplicando a métrica

mean_squared_error(valid['count'], preds) ** (1/2)
# Gerando as previsões para envio ao kaggle

test['count'] = np.exp(rf.predict(test[feats]))
# Verificando os dados

test.head()
# Gerando o arquivo para submissão ao Kaggle

test[['datetime', 'count']].to_csv('rf.csv', index=False)
# Vamos treinar outro modelo modificando os parâmetros do Randon Forest



# Aumentando a floresta -> 200 árvores

# Trabalhar a divisão dos nós -> min_samples_leaf (qtde de ocorrências em cada folha da árvore)



# Instanciando o novo modelo

rf2 = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1, min_samples_leaf=4)



# Treinando o modelo

rf2.fit(train[feats], train['count'])



# Fazendo previsões com o modelo

preds2 = rf2.predict(valid[feats])



# Aplicando a métrica

mean_squared_error(valid['count'], preds2) ** (1/2)
# Gerando as previsões para envio ao kaggle

test['count'] = np.exp(rf2.predict(test[feats]))



# Gerando o arquivo para submissão ao Kaggle

test[['datetime', 'count']].to_csv('rf2.csv', index=False)
# Vamos agora tentar reproduzir nos dados de treino o comportamento dos dados de teste

# Sequenciando a divisão de treino e validação

# Os dados de treino serão do dia 01 ao dia 15

# E os dados de validação do dia 16 ao dia 19



# Dividindo em treino e validação

train = df[df['day'] <= 15]

valid = df[df['day'] > 15]



train.shape, valid.shape
# Instanciando o novo modelo

rf3 = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1, min_samples_leaf=4)



# Treinando o modelo

rf3.fit(train[feats], train['count'])



# Fazendo previsões com o modelo

preds3 = rf3.predict(valid[feats])



# Aplicando a métrica

mean_squared_error(valid['count'], preds3) ** (1/2)
# Gerando as previsões para envio ao kaggle

test['count'] = np.exp(rf3.predict(test[feats]))



# Gerando o arquivo para submissão ao Kaggle

test[['datetime', 'count']].to_csv('rf3.csv', index=False)
# Vamos tentar melhorar o desempenho do modelo fazendo Feature Engneering

# Vamos criar colunas para representar a média de temperatura, sensação térmica e umidade

# das horas anteriores
# Juntando os dataframes

df_total = df.append(test)



# Ordenando o dataframe

df_total.sort_values('datetime', inplace=True)
# Criando a coluna rolling_temp

df_total['rolling_temp'] = df_total['temp'].rolling(3, min_periods=1).mean()



# Criando a coluna rolling_atemp

df_total['rolling_atemp'] = df_total['atemp'].rolling(3, min_periods=1).mean()



# Criando a coluna rolling_humidity

df_total['rolling_humidity'] = df_total['humidity'].rolling(3, min_periods=1).mean()



# Criando a coluna windspeed

df_total['rolling_windspeed'] = df_total['windspeed'].rolling(3, min_periods=1).mean()
# Verificando os dados

df_total.head()
# Separando os dataframes



# Dataframe de teste

test = df_total[df_total['casual'].isnull()]



# Dataframe de treino

df = df_total[~df_total['casual'].isnull()]



df.shape, test.shape
# Dividindo em treino e validação

train = df[df['day'] <= 15]

valid = df[df['day'] > 15]



train.shape, valid.shape
# Instanciando o novo modelo

rf4 = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1, min_samples_leaf=4)



# Treinando o modelo

rf4.fit(train[feats], train['count'])



# Fazendo previsões com o modelo

preds4 = rf4.predict(valid[feats])



# Aplicando a métrica

mean_squared_error(valid['count'], preds4) ** (1/2)
# Gerando as previsões para envio ao kaggle

test['count'] = np.exp(rf4.predict(test[feats]))



# Gerando o arquivo para submissão ao Kaggle

test[['datetime', 'count']].to_csv('rf4.csv', index=False)