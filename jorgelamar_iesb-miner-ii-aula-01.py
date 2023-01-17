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
# Carregar os dados (parse_dates - transporma as datas pra mim no formato certo)

train = pd.read_csv('../input/train.csv', parse_dates=[0])



test = pd.read_csv('../input/test.csv')
train.shape, test.shape
# Verificando quantidades e tipos

train.info()
#Os dados estão escondidos pelo Kaggle no arquivo de test. para depois a gente colocar

test.info()
# Verificar o dataframe

train.head().T
test.head().T
# Transformar a coluna datetime dos dados de teste - Outra forma. Se ele não fez tem algo errado, tem sujeira 

test['datetime']= pd.to_datetime(test['datetime'])



test.info()
# converter a coluna 'count'

train['count'] = np.log(train['count'])
# vamos jubntar o dados de teste e treino para facilitar as transformações

df = train.append(test)
df.tail().T
# Ordenar os dataframe pelo datetime

df.sort_values('datetime', inplace=True)
# Criando Colunas para data e hora

df['hour'] = df['datetime'].dt.hour

df['day'] = df['datetime'].dt.day

df['dayofweek'] = df['datetime'].dt.dayofweek

df['month'] = df['datetime'].dt.month

df['year'] = df['datetime'].dt.year
df
# Coluna com a diferença de temperatura

df['diff_temp'] = df['atemp'] - df['temp']
# Coluna com a temperatura da hora anterior

df['temp_shift_1'] = df['temp'].shift(1)

df['temp_shift_2'] = df['temp'].shift(2)



# Para fazer outra forma



#for i in range(3):

#    df[f'temp_shift_{i}'] = df['temp'].shift(i)



# Coluna com a sensação térmica da hora anterior

df['atemp_shift_1'] = df['atemp'].shift(1)

df['atemp_shift_2'] = df['atemp'].shift(2)
# Coluna com a diferença de temperatura da hora anterior

df['diff_temp_shift_1'] = df['diff_temp'].shift(1)

df['diff_temp_shift_2'] = df['diff_temp'].shift(2)
df.head().T
# Criando uma coluna com a media da temperatura das ultimas 4 horas

df['rolling_temp'] = df['temp'].rolling(4, min_periods=1).mean()
#Verificando o resultado

df[['temp','rolling_temp']].head(10)
df
# Separandop os dataframes

#train = df[~df['count'].isnull()]

#test = df[df['count'].isnull()]



#train.shape, test.shape
# usando outra coluna

train = df[~df['registered'].isnull()]

test = df[df['registered'].isnull()]



train.shape, test.shape
# Salva os dados do dataframe de treino

train_raw = train.copy()
# Separa os dados de treino em treino e validação

from sklearn.model_selection import train_test_split



train, valid = train_test_split(train, random_state=42)



train.shape, valid.shape
# Separando as features e o resultado

removed_cols = ['count', 'casual', 'registered', 'datetime']
# #Separando as colunas a srem usadas para treino

cols = []

for c in train.columns:

    if c not in removed_cols:

        cols.append(c)

        

cols 
#Separando as colunas a srem usadas para treino - a mesma coisa em cima 

feats = [c for c in train.columns if c not in removed_cols]



feats
# Importar os modelos

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR 
# Dicionario de modelos

models = {'RandomForest': RandomForestRegressor (random_state=42),

          'ExtraTrees': ExtraTreesRegressor (random_state=42),

          'GradientBoosting': GradientBoostingRegressor (random_state=42),

          'DecisionTree': DecisionTreeRegressor (random_state=42),

          'AdaBoost': AdaBoostRegressor (random_state=42),

          'KNM 11': KNeighborsRegressor (n_neighbors=11),

          'SVR': SVR(),

          'linear Regression': LinearRegression()}

    
#impotando metrica

# from sklearn.metrics import mean_squared_log_error Deu erro

from sklearn.metrics import mean_squared_error
# função para treino dos modelos

def run_model(model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return mean_squared_error(valid[y_name], preds) ** (1/2)
# Executando os modelos

scores = []

for name, model in models.items():

    score = run_model(model, train.fillna(-1), valid.fillna(-1), feats, 'count')

    scores.append(score)

    print(name, ':', score)