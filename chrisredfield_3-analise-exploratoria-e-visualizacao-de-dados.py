# Importar os dados novamente...

import pandas as pd

item = pd.read_csv('../input/itemLicitacao.csv')

print(item.shape)

item.head()
# Mostra (quantidade de linhas, quantidade de colunas)

item.describe()
# Descreve os indices

item.index
# Descreve as colunas

item.columns
# Descreve o DataFrame

item.info()
# Mostra a qtd de linhas

item.count()
item.hist(figsize=(15,15),bins=14);
item.qtdItem.value_counts()
item.valorItem = item.valorItem.str.replace(",",'.').astype(float)
import seaborn as sns

import matplotlib.pyplot as plt



corr = item.corr()

fig, ax = plt.subplots(figsize=(11,11))   



sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
# 1. importe o dataset de licitacao

df_lic = pd.read_csv('../input/licitacao.csv')

                     

# mostre a quantidade de linhas e colunas, índices, descreva suas colunas, descreva o dataframe

print(df_lic.shape)

df_lic.head()
df_lic.index
df_lic.dtypes
df_lic.isna().sum()
df_lic.hist(figsize=(15,15),bins=14);
# 1. Qual a soma de valores de quantidade de itens?

print("soma de valores de quantidade de itens:", item.qtdItem.sum())



# 2. Qual a diferença entre o resultado de sum() e cumsum() ?

print('diferença entre o resultado de sum() e cumsum():','cumsum mostra a soma cumulativa dentro da serie, sum somente soma todos os itens')



# 3. Qual o valor mínimo de quantidade de itens?

print('valor mínimo de quantidade de itens:',item.qtdItem.min())



# 4. Qual a maior quantidade de itens?

print('maior quantidade de itens:',item.qtdItem.max())



# 5. Qual a média da quantidade de itens?

print('média da quantidade de itens:',round(item.qtdItem.mean(),2))



# 6. Qual a mediana da quantidade de itens?

print('mediana da quantidade de itens:',item.qtdItem.median())



# 7. Qual o desvio padrão da quantidade de itens?

print('desvio padrão da quantidade de itens:', round(item.qtdItem.std(),2))
# Extra: Como desabilitar Notacao Científica

pd.set_option('display.float_format', '{:.2f}'.format)



# Descreve os Dados

item.describe()
# Exemplo Simplificado: Vamos calcular o sumário deste dataset com 5 valores.

a = pd.DataFrame({'a':[10,15,20,30,60]})

a.head()
# Diferença entre mediana e média

a.median() # mediana (o valor central) = 20

a.sum() # 135

a.mean() # media = soma dos valores / quantidade = 135 / 5 = 27



# Variancia 

# (mede o quanto cada valor está distante da média) = abs(x1 - media)**2 + abs(x2 - media)**2 .. / n - 1

# (abs(10-27)**2 + abs(15 - 27)**2 + abs(20 - 27)**2 + abs(30 - 27)**2 + abs(60 - 27)**2) / (5 - 1) = 395

a.var() 



# STD - Standard Deviation ou Desvio Padrão

# Mede o grau de dispersão dos dados. Indica o quanto um conjunto de dados é uniforme.

# Qto mais proximo de 0, mais homogeneo. Por exemplo, o desvio padrão de [1,1,1] = 0

import math

math.sqrt(395)



# Sumário

a.describe()
# Boxplot: Para visualizar os quartis, a mediana e os extremos (outliers)

a.boxplot()
# Importar uma biblioteca

import pandas as pd

print("Pandas tem o tipo {}".format(type(pd)))
# Importando módulos específicos das bibliotecas

from math import log, pi

from numpy import asarray
# Lembre que você pode sempre chamar um help() para ver a documentação, inclusive de bibliotecas

help(pd)
# E pode querer relembrar qual o tipo 

type(pd)
# Importe a biblioteca matplotlib e veja sua documentação.

help(plt)
# Importa dados de licitacoes

import pandas as pd

licit = pd.read_csv('../input/licitacao.csv')
# Licitacoes por Modalidade

licit.modalidade.value_counts()
# Gráfico de Licitações por Modalidade

import seaborn as sns

sns.countplot(y='modalidade', data=licit)
licit.situacao.value_counts().plot.bar(figsize=(18,8));
# 1. RESPOSTA  Mostre a estrutura de dados do dataset de licitacoes

import pandas as pd

licit = pd.read_csv('../input/licitacao.csv')

licit.shape

licit.index

licit.columns

licit.info()

licit.count()
# 2. Para o dataset de item, gere os sumários da quantidade de itens

item.qtdItem.sum()

item.qtdItem.cumsum()

item.qtdItem.min()

item.qtdItem.min()

item.qtdItem.max()

item.qtdItem.describe()
# 3. RESPOSTA

# Importe a biblioteca matplotlib e veja sua documentação.

import matplotlib

help(matplotlib)
# 4. Crie um gráfico para mostrar a quantidade de licitacoes por situacao

sns.countplot(y='situacao', data=licit)