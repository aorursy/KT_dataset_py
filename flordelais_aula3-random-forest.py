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
# importando os arquivos

df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df.shape, test.shape
# verificando o df de treino

df.info()
test.info()
# vamos transformar os dados. toda a transformação deverá ser replicada nos dados de teste



# aplicar log na variável de resposta



df['count'] = np.log(df['count'])
#apendando os dois para poder fazer a transformação de uma dez só. depois separa

df = df.append(test)
#convertendo a coluna datetime



df['datetime'] = pd.to_datetime(df['datetime'])
#criando novas colunas usando data e hora

df['year'] = df['datetime'].dt.year

df['month'] = df['datetime'].dt.month

df['day'] = df['datetime'].dt.day

df['hour'] = df['datetime'].dt.hour

df['dayofweek'] = df['datetime'].dt.dayofweek
df.head()
# dividir os dados que foram juntados para a transformação - treino e teste. se estiver dado nulo nas tres variáveis target

# pertence ao df de teste



# primeiro os dados de teste

test = df[df['count'].isnull()]
# agora os dados de treino o sinal de til é a negação quando uma comparação não está envolvida



df = df[~df['count'].isnull()]
df.shape, test.shape
# dividindo o df de treino



# importando o scikitlearn para a divisão da base



from sklearn.model_selection import train_test_split
#dividir 75% treino e 25% validação - padrão

train, valid = train_test_split(df, random_state=42)

train.shape, valid.shape
#escolher as colunas que vão ser usadas e as que não



# lista das colunas não usadas

removed_cols = ['casual', 'registered', 'count', 'datetime']



#lista das columas de entrada



feats = [c for c in train.columns if c not in removed_cols]

feats
# usando o random forest



# importando o modelo



from sklearn.ensemble import RandomForestRegressor
# instanciar o modelo

rf = RandomForestRegressor(random_state=42,n_jobs=-1)

#n_jobs nr de job que rodam e paralalo para dar o fit. -1 é #para usar todos os processadores

#n_estimator  - nro de arvores. o defalt é 10 mas mudará para 100 na proxima versão 0.22

#treinar o modelo com os dados de treino



rf.fit(train[feats], train['count'])
#faznedo as previsões em cima dos dados de validação

preds = rf.predict(valid[feats])
preds
from sklearn.metrics import mean_squared_error
#aplicando a métrica

mean_squared_error(valid['count'], preds) ** (1/2)

train_preds = rf.predict(train[feats])
# aplicando nos dados de treino

#dados conhecidos



mean_squared_error(train['count'], train_preds) ** (1/2)


test['count'] = np.exp(rf.predict(test[feats]))
test.head()
test[['datetime', 'count']].head()
#visualizando o arquivo para envio



test[['datetime', 'count']].to_csv('rf.csv', index=False)
test.head()
#Aumentar a floresta - 200 árvores

# instanciar o modelo

rf2 = RandomForestRegressor(random_state=42,n_jobs=-1, n_estimators=200, min_samples_leaf=5)

#n_jobs nr de job que rodam e paralalo para dar o fit. -1 é #para usar todos os processadores

#n_estimator  - nro de arvores. o defalt é 10 mas mudará para 100 na proxima versão 0.22



#treinar o modelo com os dados de treino



rf2.fit(train[feats], train['count'])



#Previsões com os dados de validação

preds2 = rf2.predict(valid[feats])



# Aplicar métrica sobre os dados de validação



mean_squared_error(valid['count'], preds2)**(1/2)
#Gerando o novo arquivo

test['count'] = np.exp(rf2.predict(test[feats]))



#visualizando o arquivo para envio

test[['datetime', 'count']].to_csv('rf2.csv', index=False)
#Pandas Rolling



df = df.append(test)
#ordenando o dataframe



df.sort_values('datetime', inplace=True)
#Criando a coluna  rolling_temp

df['rolling_temp'] = df['temp'].rolling(3,min_periods=1).mean()
#Criando a coluna  rolling_atemp

df['rolling_atemp'] = df['atemp'].rolling(3,min_periods=1).mean()
#Separando os dataframes



test = df[df['casual'].isnull()]



df = df[~df['casual'].isnull()]
# Dividindo os dados de treino em train and validation



train, valid = train_test_split(df, random_state=42)
#escolher as colunas que vão ser usadas 



#lista das columas de entrada



feats = [c for c in train.columns if c not in removed_cols]
#Novo modelo usando colunas roling

# instanciar o modelo

rf3 = RandomForestRegressor(random_state=42,n_jobs=-1, n_estimators=200, min_samples_leaf=5)

#n_jobs nr de job que rodam e paralalo para dar o fit. -1 é #para usar todos os processadores

#n_estimator  - nro de arvores. o defalt é 10 mas mudará para 100 na proxima versão 0.22



#treinar o modelo com os dados de treino



rf3.fit(train[feats], train['count'])



#Previsões com os dados de validação

preds3 = rf3.predict(valid[feats])



# Aplicar métrica sobre os dados de validação



mean_squared_error(valid['count'], preds3)**(1/2)
#Gerando o novo arquivo

test['count'] = np.exp(rf3.predict(test[feats]))



#visualizando o arquivo para envio

test[['datetime', 'count']].to_csv('rf3.csv', index=False)