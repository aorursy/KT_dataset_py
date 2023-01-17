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
# Carregando os dados

treino = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

teste = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
# Verificando o dataframe de treino

treino.info()
teste.info()
# Importando a biblioteca

from sklearn.model_selection import train_test_split
# Dividindo o datagframe

# Por padrão a divisão é um 75% para treino e 25% para validação

treino, validacao = train_test_split(treino, random_state=42)
treino.shape, validacao.shape
# Lista das colunas nao usadas

nao_usadas = ['datetime', 'casual', 'registered', 'count']
usadas = [c for c in treino.columns if c not in nao_usadas]
#importando o modelo

from sklearn.tree import DecisionTreeRegressor
ad = DecisionTreeRegressor(random_state=42)
# Treinando o modelo

#Informar as colunas de entrada e a coluna de resposta(target)

ad.fit(treino[usadas], treino['count'])
# Vamos prever com os dados de validação

previsao = ad.predict(validacao[usadas])

previsao
#Vamos olhar os dados verdadeiros de alugueis no dataframe de validacao

validacao['count']
from sklearn.metrics import mean_squared_log_error
# Calculando a metrica

mean_squared_log_error(validacao['count'], previsao**(1/2))
from sklearn.metrics import mean_squared_error

mean_squared_error(validacao['count'], previsao**(1/2))