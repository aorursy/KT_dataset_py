# importar pacotes necessários

import numpy as np

import pandas as pd
# definir parâmetros extras

pd.set_option('precision', 2)

pd.set_option('display.max_columns', 100)
# carregar arquivo de dados de treino

data = pd.read_csv('../input/iris-train.csv', index_col='Id')



# mostrar alguns exemplos de registros

data.head()
# quantas linhas e colunas existem?

data.shape
# quais são as colunas e respectivos tipos de dados?

data.info()
# sumário estatístico das características numéricas

data.describe()
# existem colunas com dados nulos?

data[data.columns[data.isnull().any()]].isnull().sum()
# quais as correlações entre as características numéricas?

data.corr()