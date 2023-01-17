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
# Importando bibliotecas:



from pydoc import help

from scipy.stats.stats import pearsonr

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
# Carregando dataset:



df = pd.read_table('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv')
# Visualizando as 10 primeiras linhas do dataset para conferir se o carregamento se deu corretamente:



df.head(10)
# Descobrindo o tamanho do dataset, linhas x colunas:



df.shape
# Descobrindo missing values:



df.info()
# Descobrindo os nomes das colunas:



df.columns.unique()
# Dropando colunas que não serão utilizadas



df.drop(columns=['DATA INICIAL', 'DATA FINAL', 'NÚMERO DE POSTOS PESQUISADOS','DESVIO PADRÃO REVENDA', 'PREÇO MÍNIMO REVENDA',

       'PREÇO MÁXIMO REVENDA', 'MARGEM MÉDIA REVENDA',

       'COEF DE VARIAÇÃO REVENDA', 'PREÇO MÉDIO DISTRIBUIÇÃO',

       'DESVIO PADRÃO DISTRIBUIÇÃO', 'PREÇO MÍNIMO DISTRIBUIÇÃO',

       'PREÇO MÁXIMO DISTRIBUIÇÃO', 'COEF DE VARIAÇÃO DISTRIBUIÇÃO'], inplace=True)
# Conferindo as colunas que restaram:



df.columns.unique()
# Renomendo a coluna Unnamed: 0:



df.columns = ['INDEX', 'REGIÃO', 'ESTADO', 'PRODUTO', 'UNIDADE DE MEDIDA',

       'PREÇO MÉDIO REVENDA', 'MÊS', 'ANO']

# Conhecendo os valores da coluna PRODUTOS:



df['PRODUTO'].unique()

# Primeiro passo será criar uma nova coluna, duplicando a coluna PRODUTOS e torna-la index:



df['produto index'] = df['PRODUTO']



# Segundoo passo será transformar a coluna PRODUTO em index



df.set_index('produto index', inplace=True)



# Ultimo passo dropar os dados da coluna PRODUTO:



df.drop(['GLP', 'GNV'], inplace=True)



# Verificando amostra dos dados:

df.sample(10)

# Conferindo o novo tamanho do dataset, linhas x colunas:



df.shape
# Conferindo se existem linhas duplicadas



df.duplicated()
# Setando uma nova coluna como index:



df.set_index('INDEX', inplace=True)

df.sample(15)
# Gráfico de PREÇO MÉDIO DE REVENDA por REGIÃO e por ANO:



plt.figure(figsize=(25,10))

sns.stripplot(x='REGIÃO', y='PREÇO MÉDIO REVENDA', hue='ANO', data=df, jitter= True)

plt.xticks(rotation=90)
# Gráfico de PREÇO MÉDIO DE REVENDA por PRODUTO por ANO:



plt.figure(figsize=(25,10))

sns.lineplot(x='ANO', y='PREÇO MÉDIO REVENDA', hue= 'PRODUTO', style= 'PRODUTO', data=df)

plt.xticks(rotation=90)

# Analisando se existe correlação entre as variáveis do dataset utilizando o coeficiente de Pearson: 



f, ax = plt.subplots(figsize=(25,12))

sns.heatmap(df.corr(), annot=True, fmt='.2f', linecolor='black', ax=ax, lw=.7)
# Gráfico de correlação entre ANO x PREÇO MÉDIO DE REVENDA:



plt.figure(figsize=(25,18))

sns.regplot(x="ANO", y="PREÇO MÉDIO REVENDA", data=df, x_estimator=np.mean)
#Transformar a variável REGIÃO em variável dummie para analisar as regiões em relação à média nacional:



df1 = pd.get_dummies(df,columns=['REGIÃO'])

df1.sample()
df1.columns.unique()
# Gráfico de PREÇO MÉDIO DE REVENDA NA REGIÃO_CENTRO OESTE por ANO:



plt.figure(figsize=(25,10))

sns.lineplot (x='ANO', y='PREÇO MÉDIO REVENDA', hue= 'REGIÃO_CENTRO OESTE', data=df1)

plt.xticks(rotation=90)
# Gráfico de PREÇO MÉDIO DE REVENDA NA REGIÃO_NORTE por ANO:



plt.figure(figsize=(25,10))

sns.lineplot (x='ANO', y='PREÇO MÉDIO REVENDA', hue= 'REGIÃO_NORTE', data=df1)

plt.xticks(rotation=90)
# Gráfico de PREÇO MÉDIO DE REVENDA NA REGIÃO_NORDESTE por ANO:



plt.figure(figsize=(25,10))

sns.lineplot (x='ANO', y='PREÇO MÉDIO REVENDA', hue= 'REGIÃO_NORDESTE', data=df1)

plt.xticks(rotation=90)
# Gráfico de PREÇO MÉDIO DE REVENDA NA REGIÃO_SUL por ANO:



plt.figure(figsize=(25,10))

sns.lineplot (x='ANO', y='PREÇO MÉDIO REVENDA', hue= 'REGIÃO_SUL', data=df1)

plt.xticks(rotation=90)
# Gráfico de PREÇO MÉDIO DE REVENDA NA REGIÃO_SUDESTE por ANO:



plt.figure(figsize=(25,10))

sns.lineplot (x='ANO', y='PREÇO MÉDIO REVENDA', hue= 'REGIÃO_SUDESTE', data=df1)

plt.xticks(rotation=90)
# Analizar correlação utilizando o coeficiente de Spearman entre as REGIÕES e PREÇO MÉDIO REVENDA:



f, ax = plt.subplots(figsize=(25,12))

sns.heatmap(df1[['PREÇO MÉDIO REVENDA','REGIÃO_CENTRO OESTE', 'REGIÃO_NORDESTE', 'REGIÃO_NORTE',

       'REGIÃO_SUDESTE', 'REGIÃO_SUL']].corr('spearman'), annot=True, fmt='.2f', linecolor='blue', ax=ax, lw=.7)




