# Importar os dados novamente...

import pandas as pd

item = pd.read_csv('../input/itemLicitacao.csv')

item.head()
# Mostra (quantidade de linhas, quantidade de colunas)

item.shape
# Descreve os indices (O RangeIndex pode ser criado a parte e no caso de series temporais vc pode reprogramar o step para ser de 7 em e 7 semanais.... ver isso com calma com Kafran)

item.index
# Descreve as colunas

item.columns
# Descreve o DataFrame

item.info()
# Mostra a qtd de linhas

item.count()
# 1. importe o dataset de licitacao



# mostre a quantidade de linhas e colunas, índices, descreva suas colunas, descreva o dataframe

licit = pd.read_csv('../input/licitacao.csv', sep=",")

licit.head()
licit.shape

licit.info()
licit.head()
# 1. Qual a soma de valores de quantidade de itens? 

# 2. Qual a diferença entre o resultado de sum() e cumsum() ?

# 3. Qual o valor mínimo de quantidade de itens?

# 4. Qual a maior quantidade de itens?

# 5. Qual a média da quantidade de itens?

# 6. Qual a mediana da quantidade de itens?

# 7. Qual o desvio padrão da quantidade de itens?
item.qtdItem.sum()
item.qtdItem.cumsum()
resultado = item.qtdItem.sum() - item.qtdItem.cumsum()

print(resultado)
item.qtdItem.min()

item.qtdItem.max()

item.qtdItem.mean()

item.qtdItem.median()

item.qtdItem.std()
# Extra: Como desabilitar Notacao Científica

pd.set_option('display.float_format', '{:.2f}'.format)



# Descreve os Dados

item.describe()
# Descreve os Dados de uma coluna

item.qtdItem.describe()
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
# Importe a biblioteca matplotlib e veja sua documentação. plt.

import matplotlib as plt



help(plt)
# Importa dados de licitacoes

import pandas as pd

licit = pd.read_csv('../input/licitacao.csv')
# Licitacoes por Modalidade (df.coluna.funcao())

licit.modalidade.value_counts()
# Gráfico de Licitações por Modalidade (a variável x ou y antes da modalidade ele determina se é vertical ou horizontal)

import seaborn as sns

sns.countplot(y='modalidade', data=licit);
licit.head()
sns.countplot(y='situacao', data=licit);
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