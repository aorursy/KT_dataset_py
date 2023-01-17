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
sem_filtro = pd.read_csv('/kaggle/input/base-describe/BASE DESCRIBE CSV.csv', sep=';', encoding='ISO-8859-1')

sem_filtro.count()


final = sem_filtro.loc[sem_filtro['DS_CARGO'].isin(['DEPUTADO FEDERAL','DEPUTADO ESTADUAL','SENADOR'])]

final.count()
#sem_filtro = pd.read_csv('/kaggle/input/base-describe/BASE DESCRIBE CSV.csv', sep=';', encoding='ISO-8859-1')

#cargos = ['DEPUTADO FEDERAL','DEPUTADO ESTADUAL','SENADOR']

#sem_filtro['DS_CARGO'].isin(['DEPUTADO FEDERAL','DEPUTADO ESTADUAL','SENADOR']).value_counts()
# Verificando o tamanho da base

print('Tamanho da base: ',final.shape)
# Listagem do nome das variáveis, linha a linha

for col in final.columns: 

    print(col)
# Verificação dos tipos de dados das variáveis

final.info()
final.shape
final.head()

import seaborn as sns

import matplotlib.pyplot as plt
final['SG_UF'].value_counts().reset_index()
plt.figure(figsize=(15,5))

plt.xticks(rotation=90)

plt.title('Candidatos por Partido')

sns.countplot(data=final, x='SG_PARTIDO', order=final.SG_PARTIDO.value_counts().index)
plt.figure(figsize=(15,5))

sns.countplot(data=final, x='TP_AGREMIACAO', order=final.TP_AGREMIACAO.value_counts().index)
plt.figure(figsize=(15,5))

sns.countplot(data=final, x='DS_CARGO', order=final.DS_CARGO.value_counts().index)
plt.figure(figsize=(15,5))

sns.countplot(data=final, x='DS_ESTADO_CIVIL', order=final.DS_ESTADO_CIVIL.value_counts().index)
final['DS_NACIONALIDADE'].value_counts().reset_index()
final['SG_UF_NASCIMENTO'].value_counts().reset_index()
#final['NR_IDADE_DATA_POSSE'].value_counts().reset_index()



plt.figure(figsize=(15,5))

final['NR_IDADE_DATA_POSSE'] = final['NR_IDADE_DATA_POSSE'].astype(str).astype('int')

final.NR_IDADE_DATA_POSSE.plot(kind='hist', bins=20)
# final['DS_GENERO'].value_counts().reset_index()

plt.figure(figsize=(15,5))

sns.countplot(data=final, x='DS_GENERO', order=final.DS_GENERO.value_counts().index)
#final['DS_GRAU_INSTRUCAO'].value_counts().reset_index()

plt.figure(figsize=(15,5))

plt.xticks(rotation=45)

sns.countplot(data=final, x='DS_GRAU_INSTRUCAO', order=final.DS_GRAU_INSTRUCAO.value_counts().index)
#final['DS_COR_RACA'].value_counts().reset_index()

plt.figure(figsize=(15,5))

plt.xticks(rotation=45)

sns.countplot(data=final, x='DS_COR_RACA', order=final.DS_COR_RACA.value_counts().index)
final['DS_OCUPACAO'].value_counts().reset_index()
plt.figure(figsize=(15,5))

plt.xticks(rotation=0)

sns.countplot(data=final, x='DS_SIT_TOT_TURNO', order=final.DS_SIT_TOT_TURNO.value_counts().index)
final['NM_COLIGACAO'].value_counts().reset_index()
#final['Votos'].value_counts().reset_index()

plt.figure(figsize=(15,5))

plt.xticks(rotation=0)

sns.boxplot(final['DS_SIT_TOT_TURNO'],final['Votos'])
pd.set_option('display.float_format','{:.2f}'.format)
final['ANO_ELEICAO'] = final['ANO_ELEICAO'].astype(str)

final['NR_CANDIDATO'] = final['NR_CANDIDATO'].astype(str)

final['NR_TURNO'] = final['NR_TURNO'].astype(str)
final.describe()
# Despesa por voto

plt.figure(figsize=(15,10))

sns.boxplot(final['DS_CARGO'],final['Votos'])



# Inserir o label

plt.title('Votos por Cargo')



# Rotacionar o label

plt.xticks(rotation=90)



# Ajustando escala Y

plt.locator_params(axis='y', nbins=20)



plt.show()
# plt.figure(figsize=(15,5))

# sns.pointplot(x='NR_IDADE_DATA_POSSE', y=final['Votos'].sum(),data=final, color='blue')

# plt.title('Quantidade de votos por idade')

# plt.grid(True, color='silver')



plt.figure(figsize=(15,5))

plt.bar(final['NR_IDADE_DATA_POSSE'], final['Votos'])
f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(final.corr(), annot=True, fmt='.2f', linecolor='black',ax=ax, lw=.7)