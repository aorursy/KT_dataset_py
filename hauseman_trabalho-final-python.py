# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Carregando dos dados da base finaciamento do Fies

fies=pd.read_csv('../input/FINANCIAMENTO_CONCEDIDOS_SEMESTRE_2_2011.csv', sep=';', encoding='latin1')

#Verificando valores nulos 

#fies.isnull().any()

fies.isnull().sum()
#Tratando valores nulos

fies.dropna(inplace=True)
#Verificando valores o resultado após tratar os registros 



fies.isnull().sum()
#Verificando duplicidade 

fies.duplicated()
#Colunas presentes no DataFrame

fies.columns
#Verificando número de linhas e colunas

print("Number of (rows,columns):",fies.shape)
#Verificando os 10 primeiros resgistros da tabeladf.columns

fies.head(10)
# Verificar as 5 últimas linhas

fies.tail()


fies.index
fies.dtypes
fies.info()
#Removendo colunas que não entrarão na análise

fies.drop(['NU_CNPJ_MANTENEDORA','CO_MANTENEDORA','NU_CNPJ_MANTENEDORA','CO_MUNICIPIO_IES','CO_PROCESSO', 'CO_CONTRATO_FIES',

          'CO_ADITAMENTO','CO_INSCRICAO','CO_INSCRICAO_EXT_ALUNOS', 'CO_CONTRATO_FIES_EXT_ALUNOS', 'CO_IES_EXT_ALUNOS',

          'CO_ADITAMENTO_EXT_ALUNOS','CO_PERIODICIDADE_CUR','CO_MUNICIPIO_MANTENEDORA','CO_IES','CO_AGENTE_FINANCEIRO',

           'CO_TIPO_CURSO','CO_CURSO'], axis=1,inplace=True) 
fies.columns
# Convertendo variaveis definidas como string para númerico

fies.VL_SEMESTRE = fies.VL_SEMESTRE.astype('category')

fies.VL_SEMESTRE = fies.VL_SEMESTRE.str.replace(",",".").astype(float)

fies.VL_MENSALIDADE_EXT_ALUNOS = fies.VL_MENSALIDADE_EXT_ALUNOS.astype('category')

fies.VL_MENSALIDADE_EXT_ALUNOS = fies.VL_MENSALIDADE_EXT_ALUNOS.str.replace(",",".").astype(float)

fies.VL_MENSALIDADE = fies.VL_MENSALIDADE.astype('category')

fies.VL_MENSALIDADE = fies.VL_MENSALIDADE.str.replace(",",".").astype(float)

fies.VL_REPASSE = fies.VL_REPASSE.astype('category')

fies.VL_REPASSE = fies.VL_REPASSE.str.replace(",",".").astype(float)
#Verificando a alteração do tipo de dados

fies.dtypes
# Estatísticas das variáveis

fies.describe()
# Analisando a Estatística Descritiva das variáveis VL_MENSALIDADE,VL_REPASSE, VL_SEMESTRE

fies[['VL_MENSALIDADE','VL_REPASSE','VL_SEMESTRE']].describe()
fies.boxplot(column='VL_MENSALIDADE_EXT_ALUNOS')
fies.boxplot(column='VL_REPASSE')
fies.VL_MENSALIDADE.max()

fies.VL_MENSALIDADE.min()
fies.corr()
#Matriz de correlação

f,ax = plt.subplots(figsize=(15,6))

sns.heatmap(fies.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=.7)
fies['VL_MENSALIDADE'].describe()
fies.VL_REPASSE.plot(kind = 'line', color = 'g',label = 'VL_REPASSE',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

fies.VL_MENSALIDADE.plot(color = 'r',label = 'VL_MENSALIDADE',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')             

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
fies['VL_MENSALIDADE'].value_counts()
plt.scatter(x=fies['SG_UF'],y=fies['VL_MENSALIDADE'],color='y',alpha=0.7)

plt.xlabel('SG_UF')

plt.ylabel('VL_MENSALIDADE')

plt.show()
fies[fies['VL_MENSALIDADE'] == fies['VL_MENSALIDADE'].min()]
fies[fies['VL_MENSALIDADE'] == fies['VL_MENSALIDADE'].max()]
fies.plot(kind='scatter', x='VL_MENSALIDADE', y='VL_REPASSE',alpha = 0.5,color = 'red')

plt.xlabel('VL_MENSALIDADE')              

plt.ylabel('VL_REPASSE')

plt.title('Valor repasse x Valor Mensalidade')           
fies.QT_SEMESTRE_FINANCIADO.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
fies[fies['VL_REPASSE'] == fies['VL_REPASSE'].max()]
fies[fies['VL_REPASSE'] == fies['VL_REPASSE'].min()]
fies[['DS_CURSO','VL_MENSALIDADE','SG_UF']].sample(20)
fies[['DS_CURSO','VL_REPASSE','SG_UF', 'DS_RACA_COR', 'DS_SEXO']].sample(20)
%matplotlib inline
fies['SG_UF'].value_counts()
fies.nunique()
# Quantidade de contratos financiamento por raça

fies['DS_RACA_COR'].value_counts().plot.pie()
# Quantidade de finaciamento por Sexo

fies['DS_SEXO'].value_counts().plot.bar()
# Quantidade de contratos financiados por Instituição Financeira

fies['NO_AGENTE_FINANCEIRO'].value_counts().plot.bar()
# Quantidade de contratos financiados por Instituição Financeira

fies['NO_AGENTE_FINANCEIRO'].value_counts()
# Quantidade de financiamento por UF 

fies['SG_UF'].value_counts()
# Quantidade de financiamento por UF 

fies['SG_UF'].value_counts().plot.bar()
sns.catplot(x='SG_UF',kind='count',data=fies, height=11)
sns.catplot(x='DS_ESTADO_CIVIL',kind='count',data=fies, height=11)
sns.catplot(x='DS_SEXO',kind='count',data=fies, height=11)


sns.catplot(x='ST_ENSINO_MEDIO_ESCOLA_PUBLICA',kind='count',data=fies, height=11)
 

sns.catplot(x='NU_PERCENT_SOLICITADO_FINANC',kind='count',data=fies, height=11)
# Plotando gráfico de linhas com valor da mensalidade por UF

plt.subplots(figsize=(15,6))

sns.lineplot(x='SG_UF', y='VL_MENSALIDADE', data=fies)

plt.xticks(rotation=90)

plt.title('Valor da mensalidade por UF')

plt.show()