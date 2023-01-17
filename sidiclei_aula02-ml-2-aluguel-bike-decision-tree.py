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

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv' )



train.shape, test.shape
train.head()
train.info()
test.info()
# Aplicar log no count

train['count'] = np.log(train['count'])
train.head()
df = pd.concat([train, test])
df.shape
df.dtypes
# Converter  datetime

df['datetime'] = pd.to_datetime(df['datetime'])
df.dtypes
df.iloc[10880:10900]
df.loc[2]
# Resetar o indice

df.reset_index(inplace=True)
df.loc[2]
# Transformações das datas



df['year'] = df.datetime.dt.year # df['datetime'].dt.year

df['month'] = df.datetime.dt.month

df['day'] = df.datetime.dt.day

df['dayofweek'] = df.datetime.dt.dayofweek
df['hour'] = df.datetime.dt.hour
# Criar coluna com a diferença de temperatura



df['diff_temp'] = df['atemp'] - df['temp']
# criar coluna com a temperatura da hora anterior



df['temp_shift_1'] = df['temp'].shift(1)

df['temp_shift_2'] = df['temp'].shift(2)



## Outra maneira



'''

for i in range(3)

    df[f'temp_shift_{1}'] = df['temp'].shift(i) 

'''
# criar coluna com a sensação termica da hora anterior



df['atemp_shift_1'] = df['atemp'].shift(1)

df['atemp_shift_2'] = df['atemp'].shift(2)
# criar coluna com a a diferença de temperatura da hora anterior



df['diff_shift_1'] = df['diff_temp'].shift(1)

df['diff_shift_2'] = df['diff_temp'].shift(2)




# Criar coluna com a média da temperatura das últimas 4 horas



df['rolling_temp'] = df['temp'].rolling(4,min_periods=1).mean()



# Verificando o resultado

df[['temp','rolling_temp']].head(10)
# Separar os dataframes

train = df[~df['count'].isnull()]

test = df[df['count'].isnull()]
train.shape, test.shape
# Separando o dataframe de treino em teste e validação

from sklearn.model_selection import train_test_split



train, valid = train_test_split(train, random_state=42)



train.shape, valid.shape
# Criando a lista de features



removed_col = ['count', 'casual','registered','datetime', 'index']



# Separando as colunas a serem usadas no treino

feats = [c for c in train.columns if c not in removed_col]
feats
# Usando o modelo de regressão por Árvore de Decisão



# Passo 1: Importar o modelo

from sklearn.tree import DecisionTreeRegressor

# Passo 2: Instanciar o modelo

dtr = DecisionTreeRegressor(random_state=42)
# Passo 3: Treinar o modelo

dtr.fit(train.fillna(-1)[feats],train.fillna(-1)['count'])
# Passo 4: Testar o modelo - Realizar as predições

dtr.predict(valid.fillna(-1)[feats])
from sklearn import tree

import graphviz

from IPython.display import SVG

from IPython.display import display
data = tree.export_graphviz(dtr, out_file=None, feature_names=feats,

                            class_names=['count'], filled=True,

                            rounded=True, special_characters=True, 

                            max_depth=3)
graph = graphviz.Source(data)

display(SVG(graph.pipe(format='svg')))
# Metrica de avaliação

from sklearn.metrics import mean_squared_error
preds = dtr.predict(valid.fillna(-1)[feats])
mean_squared_error(valid.fillna(-1)['count'],preds) ** (1/2)
# Tentando melhorar nosso modelo

dtr2 = DecisionTreeRegressor(random_state=42,min_samples_leaf=8)

dtr2.fit(train.fillna(-1)[feats],train.fillna(-1)['count'])

preds = dtr2.predict(valid.fillna(-1)[feats])

mean_squared_error(valid.fillna(-1)['count'],preds) ** (1/2)
# Submissao ao Kaggle

test['count'] = dtr2.predict(test[feats])

# Tirando o log

test['count'] = np.exp(test['count'])
test.head()
test[['datetime','count']].to_csv('dtr2.csv', index=False)