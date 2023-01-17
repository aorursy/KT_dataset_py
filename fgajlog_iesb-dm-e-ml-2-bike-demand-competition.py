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
#carregar os dados



train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')



teste = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
#verificando df treino



train.info()
#Verificando o df de teste

teste.info()
train.head()
#Transformando o dataframe original na coluna count

#vamos usar escala logaritimica



train['count'] = np.log(train['count'])

train = train.append(teste)
train.head()



train['datetime'] = pd.to_datetime(train['datetime'])
#crindo nova coluna usando data e hora



train['year'] = train['datetime'].dt.year

train['month'] = train['datetime'].dt.month

train['day'] = train['datetime'].dt.day

train['hour'] = train['datetime'].dt.hour

train['dayofweek'] = train['datetime'].dt.dayofweek



train.head()
#separando o df de treino e teste



#primeiro teste



teste = train[train['count'].isnull()]
teste.shape
#separando o df de treino e teste



#segundo treino



treino = train[~train['count'].isnull()]
treino.shape
#Separando o df de treino em treino/validação (def = 75/25)



from sklearn.model_selection import train_test_split



treino, validacao = train_test_split(treino, random_state=42)
print(treino.shape)

treino.head()
print(validacao.shape)

validacao.head()

#selecionando colunas



#colunas nao usadas:



nao_usadas = ['datetime', 'casual', 'registered', 'count']



#colunas usadas:



#lista_usadas = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity','windspeed']



usadas = [c for c in treino if c not in nao_usadas]



usadas
#importando



from sklearn.tree import DecisionTreeRegressor
#instanciando objeto de decision tree



ad = DecisionTreeRegressor(random_state=42)
#treinando o modelo

#informar as colunas de entrada e a coluna de resposta (target)



ad.fit(treino[usadas], treino['count'])
#prever os dados de validação



previsao = ad.predict(validacao[usadas])
previsao
validacao['count']
#usando a metrica para validar os dados



from sklearn.metrics import mean_squared_error
#Calculando a métrica

mean_squared_error(validacao['count'], previsao)**(1/2)





#1.5756091264794638

#erro alto... quanto mais perto de 0 melhor
#importar o modelo Random Forest



from sklearn.ensemble import RandomForestRegressor
#instanciar o modelo

rf = RandomForestRegressor(random_state=42, n_jobs=1)
#treinando o modelo

rf.fit(treino[usadas], treino['count'])
#Fazendo previsões em cima dos dados de validação



preds = rf.predict(validacao[usadas])
preds
#verificando o real



validacao['count'].head(3)
#verificando o modelo com relação a métrica



#importando a métrica



from sklearn.metrics import mean_squared_error
#aplicando a métrica

mean_squared_error(validacao['count'], preds) ** (1/2)

#0.348204 do professor
#vamos prever com base nos dados de treino

# como o modelo se comporta prevendo em cima de dados conhecidos

# o modelo ja esta treinado



treino_preds = rf.predict(treino[usadas])



mean_squared_error(treino['count'], treino_preds) ** (1/2)
#Gerando as previsões para envio ao Kaggle

teste['count'] = np.exp(rf.predict(teste[usadas]))
#visualizando o arquivo para envio

teste[['datetime','count']].head()
#gerando csv

teste[['datetime','count']].to_csv('rf.csv', index=False)