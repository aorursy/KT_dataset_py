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
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
# importando as bases 

treino = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv') # trino dados do dia 1 ao dia 19

teste = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv') # base de teste do dia 20 ao final do mês
treino.shape, teste.shape
display(treino.info())

display(teste.info())
# realizando as tranformações nos dados
# Aplicar log na variável de resposta

treino['count'] = np.log(treino['count'])
# Juntando os dataframes para realizar as modificações

# As observações do teste ficaram com o campo count nulo

# Concatenando as bases para realizar as transformações nas duas bases de uma vez só

treino = treino.append(teste)
# transformando o tipo da variável datetime em datetime

treino['datetime'] = pd.to_datetime(treino['datetime'])
# Criando novas colunas com a dada e hora (feature engeneering)

treino['year'] = treino['datetime'].dt.year

treino['month'] = treino['datetime'].dt.month

treino['day'] = treino['datetime'].dt.day

treino['dayofweek'] = treino['datetime'].dt.dayofweek

treino['hour'] = treino['datetime'].dt.hour

# separando so dataframes

teste = treino[treino['count'].isnull()]
teste.shape
treino = treino[~treino['count'].isnull()]
treino.shape
treino, validacao = train_test_split(treino, random_state=42)
display(treino.info())

display(validacao.info())
# selecionando as variáveis que serão utilizadas no treinamento

nao_usadas = ['casual', 'registered', 'count', 'datetime']



# Xriar a lista das colunas de entrada

usadas = [c for c in treino.columns if c not in nao_usadas]
print(usadas)
# Instanciando o modelo

random_forest = RandomForestRegressor(random_state=42, n_jobs=-1)
# Treinando o modelo

random_forest.fit(treino[usadas], treino['count'])
# Prevendo os resultados

previsao = random_forest.predict(validacao[usadas])
# Valores preditos

previsao
# Avaliando o modelo com o SRMSLE (Square Root Mean Squared Log Error)

mean_squared_error(validacao['count'], previsao)**(1/2)
# Vamos prever com base nos dados de treino

# como o modelo se comporta prevendo em cima de dados conhecidos

# Verificar se está generalizando bem, caso o erro seja zero na base de treino, é um forte sinal de overfitting



treino_preds = random_forest.predict(treino[usadas])

mean_squared_error(treino['count'], treino_preds) ** (1/2)
# Gerando as previsões para envio ao Kaggle



teste['count'] = np.exp(random_forest.predict(teste[usadas]))
# Gerando o arquivo para submeter ao kaggle

teste[['datetime', 'count']].head()
teste[['datetime', 'count']].to_csv('rf.csv' ,index=False)