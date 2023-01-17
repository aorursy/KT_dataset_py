# importar pacotes necessários

import numpy as np

import pandas as pd
# definir parâmetros extras

pd.set_option('precision', 3)

pd.set_option('display.max_columns', 100)
prefixo_arquivos = '/kaggle/input/zoo-animals-extended-dataset/'
# carregar arquivo de dados

data = pd.read_csv(prefixo_arquivos + 'zoo2.csv', index_col='animal_name')



# mostrar alguns exemplos de registros

data.head()
# quantas linhas e colunas existem?

data.shape
# carregar arquivo de dados

data2 = pd.read_csv(prefixo_arquivos + 'zoo3.csv', index_col='animal_name')



# mostrar alguns exemplos de registros

data2.head()
# quantas linhas e colunas existem?

data2.shape
# unir ambos os dados

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
data.tail()
# sumário estatístico das características numéricas

data.describe()
# quais as correlações entre as características numéricas?

data.corr()
# show variable correlation which is more than 0.7 (positive or negative)

corr = data.corr()

corr[corr != 1][abs(corr) > 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)
data.groupby('class_type').mean()
# importar pacotes necessários

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#from sklearn.utils import shuffle
# definir parâmetros extras

import warnings

warnings.filterwarnings("ignore")

sns.set(style="white", color_codes=True)
# 1-7 is Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate

animal_type = ['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']



data['class_name'] = data['class_type'].map(lambda x: animal_type[x-1])



data.iloc[:,-2:].head()
# quantos registros existem de cada espécie?

data['class_type'].value_counts()
sns.countplot(data['class_name'])
data.legs.unique()
sns.countplot(data['legs'])
# gerar mapa de calor com a correlação das características

plt.figure(figsize=(14,14))

sns.heatmap(data.corr(), annot=True, fmt='.2f')
data.groupby('class_name').mean()
g = sns.FacetGrid(data, col="class_name")

g.map(plt.hist, "legs")

plt.show()