# importar pacotes necessários

import numpy as np

import pandas as pd
# definir parâmetros extras

pd.set_option('precision', 4)

pd.set_option('display.max_columns', 100)
# carregar arquivo de dados de treino

data = pd.read_csv('../input/abalone-train.csv', index_col='id')



# mostrar alguns exemplos de registros

data.head()
# quantas linhas e colunas existem?

data.shape
# quais são as colunas e respectivos tipos de dados?

data.info()
# existem colunas com dados nulos?

data[data.columns[data.isnull().any()]].isnull().sum()
# sumário estatístico das características numéricas

data.describe().T
# quais as correlações entre as características numéricas?

data.corr()
# show variable correlation which is more than 0.6 (positive or negative)

corr = data.corr()

corr[corr != 1][abs(corr) > 0.6].dropna(how='all', axis=1).dropna(how='all', axis=0)
data.groupby('rings').mean()
numeric_feats = data.dtypes[data.dtypes != "object"].index

numeric_feats
data.head(10).T
data.isna().sum()
data.sex.describe()
data.head()