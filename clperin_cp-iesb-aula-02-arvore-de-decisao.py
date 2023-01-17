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
df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df.shape, test.shape
# Verificando os dataframes

df.info()
test.info()
# REalizando as trasnformações nos dados
# Aplicar log na variável de resposta

df['count'] = np.log(df['count'])
# Juntando os dataframente

df = df.append(test)

df.info()
# Converter a coluna datetime

df['datetime'] = pd.to_datetime(df['datetime'])
# Criar Future Engineering(criar colunas) para o datetime

df['year'] = df['datetime'].dt.year

df['month'] = df['datetime'].dt.month

df['day'] = df['datetime'].dt.day

df['dayofweek'] = df['datetime'].dt.dayofweek

df['hour'] = df['datetime'].dt.hour
# Separando os dataframes

test = df[df['count'].isnull()]
# Excluindo os nulos do dataframe df

df = df[~df['count'].isnull()]
df.shape, test.shape
# Dividindo o dataframe de treino

# Importando o método scikitlearn para divisão

from sklearn.model_selection import train_test_split
#Dividir a base de treino

train, valid = train_test_split(df, random_state=42)
# verificando tamanhos

train.shape, valid.shape
# Selecionando as colunas que iremos usar como entrada

# lista das colunas não usadas

removed_cols = ['casual', 'registered', 'count', 'datetime']



# Criar a lista das colunas de entrada

feats = [c for c in train.columns if c not in removed_cols]
#Usando o modelo random forest

# Importando o modelo

from sklearn.ensemble import RandomForestRegressor
#Instanciar o modelo

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
# Treinar o modelo. Romário não gostaria disso.

rf.fit(train[feats], train['count'])
# Fazendo predições em cima dos dados de validação

preds = rf.predict(valid[feats])
# verificando as previsões

preds
# Verificando o real

valid['count'].head(3)
# Vamos verificar o modelo com relação à métrica



# Importar a métrica

from sklearn.metrics import mean_squared_error
# Aplicando a métrica

mean_squared_error(valid['count'], preds)**(1/2)
# Vamos prever com base nos dados de treino

# Como o modelo se comporta prevendo em cima de dados desconhecidos?

train_preds = rf.predict(train[feats])



mean_squared_error(train['count'], train_preds)**(1/2)

#Gerando as previsões para envio ao Kaggle
# vamos fazer previsões para a base de teste

test['count'] = np.exp(rf.predict(test[feats]))



#mean_squared_error(test['count'], test_preds)**(1/2)

# Gerando o arquivo para submeter ao kaggle

test[['datetime', 'count']].head(50)
test[['datetime', 'count']].to_csv('rf.csv', index=False)