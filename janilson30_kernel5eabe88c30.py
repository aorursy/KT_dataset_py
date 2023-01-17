## Centro Universitário IESB 

### Pós Graduação em Ciência de Dados

### Disciplina - Data Mining e Machine Learning II

### Turma Asa Sul (Abril/2020)

### Professor Marcos Guimarães

### Aluno Janilson da Silva Nascimento (1931133050)



## Objetivo:

###### manter percentual de inadimplência baixo, identificando os possíveis maus pagadores.

###### A base de dados "Home Equity" dispòe dados de empréstimo e 12 variáveis observadas. 

###### A variável alvo (BAD) indica o valor 1 quando o cliente não pagou e valor 0, quando honra o pagamento.

###### Modelos: Random Forest Classifier, XGBosst e XGBoost 



## Dicionário de Dados

###### LOAN   : Montante de pedido requisitado

###### MORTDUE: Valor devido da hipoteca existente

###### VALUE  : Valor da garantia 

###### REASON : DebtCon = consolidação do débito / HomeImp = melhoria na casa

###### JOB    : categorias ocupacionais 

###### YOJ    : tempo em ano no emprego atual 

###### DEROG  : números dos principais relatórios depreciativos 

###### DELINQ : número de linhas de créditos inadimplentes

###### CLAGE  : idade da linha comercial mais antiga em meses

###### NINQ   : número de linhas de crédito recentes

###### CLNO   : números de linhas de crédito 

###### DEBTINC: Tation para empréstimos 

###### BAD    : 1 = Cliente Inadimplente e 0 - Clente Adimplente 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Bibliotecas

import numpy as np 

import pandas as pd 

import math

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from scipy.stats import ttest_ind

%matplotlib inline



# Diretórios

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# abertura do arquivo, formato CSV, a ser utilizado no modelo 

df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

df_copia = df.copy()
# mostra registro aleatorio, estrutura dos campos e total de linhas e colunas  

display(df.sample())

df.info()

df.shape
# Permitie realizar análise descritiva

df.describe(include='all')
# mostra as variáveis com valores nulos

MissingValues = df.isnull().sum().rename_axis('Colunas').reset_index(name='Missing Values')

MissingValues
# mostra as médias estatísticas das variáveis númericas 

df.describe()
# imputação dos valores das variáveis númericas pela média #

df.fillna(df.mean(), inplace=True)
# histograma dos dados das variáveis descritivas, considerando os missing

print(df[df['REASON'].isnull()]['VALUE'].hist(bins=10))

print(df[df['JOB'].isnull()]['VALUE'].hist(bins=10))
# Imputando conteúdo 'Other', em substituição dos missing na variável JOB

df.update(df['JOB'].fillna('Other'))
# Dropando os registros com conteudo vazio na variável REASON

df = df.dropna(subset=['REASON'])
# mostra novamente as variáveis com valores nulos

MissingValues = df.isnull().sum().rename_axis('Colunas').reset_index(name='Missing Values')

MissingValues
# mostra a distribuição dos dados das variáveis númericas

sns.pairplot(df)
# mostra a soma por variaveis agrupados pela variável descritiva JOB

df.groupby("JOB").count()
# mostra a soma por variaveis agrupados pela variável descritiva REASON

df.groupby("REASON").count()
# Verifica o balanceamento entre adimplentes (0) e inadimplentes (1)

df['BAD'].value_counts().plot(kind='bar')
# modelos utilizados: RandomForestClassifier, cross_val_score, XGBoost e GradientBoostingClassifier. 

# objetivo - criar modelo capaz de ajudar na tomada de decisão sobre a concessão do crédito.

# Bibliotecas  ligadas aos modelos 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import classification_report
# Dividindo o DataFrame

from sklearn.model_selection import train_test_split