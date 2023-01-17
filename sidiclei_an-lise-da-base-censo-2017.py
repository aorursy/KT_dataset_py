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
# Bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Biblioteca para gráficos interativos

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import cufflinks as cf

cf.go_offline()

# Para Notebooks

init_notebook_mode(connected=True)

aluno = pd.read_csv('../input/DM_ALUNO.csv', sep='|', nrows=1000, parse_dates=['DT_INGRESSO_CURSO'])

docente = pd.read_csv('../input/DM_DOCENTE.csv', sep='|', nrows=1000)

curso = pd.read_csv('../input/DM_CURSO.csv', sep='|', nrows=1000, encoding='latin1' )

local_oferta = pd.read_csv('../input/DM_LOCAL_OFERTA.csv', sep='|', encoding='latin1')

ies = pd.read_csv('../input/DM_IES.csv', sep='|', encoding='latin1')

aux_area = pd.read_csv('../input/TB_AUX_AREA_OCDE.csv', sep='|', encoding='latin1')
# Verificando a quantidade de colunas por TIPO

aluno.dtypes.value_counts()
curso.dtypes.value_counts()
docente.dtypes.value_counts()
ies.dtypes.value_counts()
local_oferta.dtypes.value_counts()
aux_area.dtypes.value_counts()
# Função para montar uma lista com as colunas e tipos do HIVE

def listaColunas(tabela):

    lista = []

    for col in tabela.columns:

        if tabela[col].dtype == 'object':

            tipo = col + ' STRING,'

        if tabela[col].dtype == 'int64':

            tipo = col + ' INT,'

        if tabela[col].dtype == 'float64':

            tipo = col + ' FLOAT,'

        if tabela[col].dtype == 'datetime64[ns]':

            tipo = col + ' TIMESTAMP,'

        lista.append(tipo) 

    return lista 
# Usando join para montar uma STRING sem as 'aspas simples' das COLUNAS TIPO

' '.join(listaColunas(aluno))
# Deletando todas as colunas que possui preenchimento com NaN

aluno.dropna(axis='columns', inplace=True)
aluno.columns
aluno.head(5).T
aluno.info()
# Usando join para montar uma STRING sem as 'aspas simples' das COLUNAS TIPO

' '.join(listaColunas(docente))
# Deletando todas as colunas que possui preenchimento com NaN

docente.dropna(axis='columns', inplace=True)
docente.columns
docente.head(5).T
docente.info()
# Usando join para montar uma STRING sem as 'aspas simples' das COLUNAS TIPO

' '.join(listaColunas(curso))
# Deletando todas as colunas que possui preenchimento com NaN

curso.dropna(axis='columns', inplace=True)
curso.columns
curso.head().T
curso.info()
# Usando join para montar uma STRING sem as 'aspas simples' das COLUNAS TIPO

' '.join(listaColunas(local_oferta))
# Deletando todas as colunas que possui preenchimento com NaN

local_oferta.dropna(axis='columns', inplace=True)
local_oferta.head().T
local_oferta.info()
# Usando join para montar uma STRING sem as 'aspas simples' das COLUNAS TIPO

' '.join(listaColunas(ies))
ies.info()
ies.head(3).T
# Usando join para montar uma STRING sem as 'aspas simples' das COLUNAS TIPO

' '.join(listaColunas(aux_area))
aux_area.head().T
aux_area.info()
# Verificando se todas as colunas do tipo 'int64' estão preenchidas

curso.select_dtypes('int64').info()
# Verificando se todas as colunas do tipo 'float64' estão preenchidas

curso.select_dtypes('float64').info()
# Verificando se todas as colunas do tipo 'datetime64[ns]' estão preenchidas

aluno.select_dtypes('datetime64[ns]').info()
# Verificando se todas as colunas do tipo 'object' estão preenchidas

curso.select_dtypes('object').info()
# Os 5 Alunos mais velhos

aluno.nlargest(5,'NU_IDADE')
# Os 5 Alunos mais novos

aluno.nsmallest(5,'NU_IDADE')
# Visualização das estatisticas das variáveis

aluno.describe()
# Verificando as colunas que permaneceram no dataframe

aluno.columns
# Verificando a quantidade de colunas por TIPO

aluno.dtypes.value_counts()
aluno.to_csv('aluno.csv', index=False)
curso.info()
# Visualização em PIZZA da quantidade total de ALUNOS por SEXO

aluno.iplot(kind='pie', labels='TP_SEXO',values='NU_IDADE')



# Visualizando o histograma dos alunos pela IDADE

plt.figure(figsize=(15,10))

sns.distplot(aluno['NU_IDADE'], kde=True)
aluno['NU_IDADE'].iplot(kind='hist')
# Apresentando o histograma de ALUNOS por CURSO

plt.figure(figsize=(15,10))

sns.distplot(aluno['CO_CURSO'], kde=True)
# Apresentando o histograma de ALUNOS por CURSO

aluno['CO_CURSO'].iplot(kind='hist')
# Apresentando a quantidade total de alunos por SEXO

plt.figure(figsize=(16,4))

sns.countplot(x='TP_SEXO', data=aluno)
# Apresentando a quantidade de ALUNOS por CURSO

plt.figure(figsize=(16,4))

sns.countplot(x='CO_CURSO', data=aluno)
# Apresentado o número de ALUNOS por CURSO agrupados por SEXO

plt.figure(figsize=(16,8))

sns.countplot(x='CO_CURSO', data=aluno, hue='TP_SEXO')

# Apresentando o número de ALUNOS por GRAU ACADEMICO

plt.figure(figsize=(16,4))

sns.countplot(x='TP_GRAU_ACADEMICO', data=aluno)

# Apresentado o número de ALUNOS por GRAU ACADEMICO agrupados por SEXO

plt.figure(figsize=(16,8))

sns.countplot(x='TP_GRAU_ACADEMICO', data=aluno, hue='TP_SEXO')

# Apresentado o número de ALUNOS por GRAU ACADEMICO agrupados por COR/RAÇA

plt.figure(figsize=(16,8))

sns.countplot(x='TP_GRAU_ACADEMICO', data=aluno, hue='TP_COR_RACA')

# Apresentado o número de alunos por tipo de COR/RAÇA agrupados por SEXO

plt.figure(figsize=(16,8))

sns.countplot(x='TP_COR_RACA', data=aluno, hue='TP_SEXO')

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score