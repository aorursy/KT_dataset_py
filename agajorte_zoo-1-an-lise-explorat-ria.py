# importar pacotes necessários

import numpy as np

import pandas as pd
# definir parâmetros extras

pd.set_option('precision', 3)

pd.set_option('display.max_columns', 100)
# carregar arquivo de dados de treino

data = pd.read_csv('../input/zoo-train.csv', index_col='animal_name')



# mostrar alguns exemplos de registros

data.head()
# quantas linhas e colunas existem?

data.shape
# carregar arquivo de dados de treino

data2 = pd.read_csv('../input/zoo-train2.csv', index_col='animal_name')



# mostrar alguns exemplos de registros

data2.head()
# quantas linhas e colunas existem?

data2.shape
# unir ambos os dados de treinamento

data = data.append(data2)



# mostrar tamanho

print(data.shape)



# mostrar alguns exemplos de registros

data.tail()
# quais são as colunas e respectivos tipos de dados?

data.info()
# existem colunas com dados nulos?

data[data.columns[data.isnull().any()]].isnull().sum()
# classe do animal deve ser uma categoria

data['class_type'] = data['class_type'].astype('category')
# atributos devem ser convertidos para 0 e 1



objcols = data.select_dtypes(['object']).columns

print(objcols)



data[objcols] = data[objcols].astype('category')

for col in objcols:

    data[col] = data[col].cat.codes
data.info()
data.tail()
# sumário estatístico das características numéricas

data.describe()
# quais as correlações entre as características numéricas?

data.corr()
# show variable correlation which is more than 0.7 (positive or negative)

corr = data.corr()

corr[corr != 1][abs(corr) > 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)
data.groupby('class_type').mean()
# gravar arquivo CSV consolidado

#data.to_csv('zoo-train-all.csv')
# carregar arquivo de dados de treino

#data = pd.read_csv('zoo-train-all.csv', index_col='animal_name')

#data['class_type'] = data['class_type'].astype('category')



# mostrar alguns exemplos de registros

data.sample()