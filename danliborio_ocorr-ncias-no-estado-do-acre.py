# importanto as principais bibliotecas que iremos utilizar

import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime as dt

# apresentando o diretório onde estão salvos os arquivos

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Leitura das bases de dados
Ano_2018 = pd.read_csv("/kaggle/input/ocorrencias/Base_ocorrencias_2018.csv", sep=';',  encoding='latin1')
Ano_2019 = pd.read_csv("/kaggle/input/ocorrencias/Base_ocorrencias_2019.csv", sep=';',  encoding='latin1')
Ano_2020 = pd.read_csv("/kaggle/input/ocorrencias/Base_ocorrencias_2020.csv", sep=';',  encoding='latin1')
# visualizando se os arquivos têm as mesmas colunas

print('2018', Ano_2018.columns)
print('2019', Ano_2019.columns)
print('2020', Ano_2020.columns)
# concatenando as bases para um unico arquivo
df = pd.concat([Ano_2018,Ano_2019,Ano_2020])
df.head()
# primeiramente iremos conhecer a quantidade de linhas e colunas estaremos trabalhando, informação necessária caso indentificamos algum campo sem preenchimento que podemos excluir do nosso dataset
print('total de linha do nosso dataset:', df.shape)
df.info()
# Visualizando as linhas duplicadas do nosso dataset

df[df.duplicated()== True]
#dropando os registros duplicados na base

df = df.drop_duplicates()
# visualizando os campos que estão sem informação
df.isnull().sum()
# dropando a coluna "NO_RODOVIA" pois 99,66% da coluna está sem registro de informação

df = df.drop('NO_RODOVIA', axis=1)
# Em nosso dataset, temos um campo com informações de data e hora. Vamos converter o mesmo para que possamos aproveitar melhor as informações dos dados.
df['DT_REGISTRO_OCORRENCIA'] = pd.to_datetime(df['DT_REGISTRO_OCORRENCIA'])

df.dtypes
# Criando as variávies dia, mes, ano, dia da semana, 
df['ANO'] = df['DT_REGISTRO_OCORRENCIA'].dt.year
df['MES'] = df['DT_REGISTRO_OCORRENCIA'].dt.month
df['DIA'] = df['DT_REGISTRO_OCORRENCIA'].dt.day
df['DIA_DA_SEMANA'] = df['DT_REGISTRO_OCORRENCIA'].dt.dayofweek
##print(df.columns)

df.groupby(df['TIPO_LOCAL'])['TIPO_LOCAL'].count().plot().barh()