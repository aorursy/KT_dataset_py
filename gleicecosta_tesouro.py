# Este ambiente Python 3 vem com muitas bibliotecas úteis de análise instaladas

# É definido pela imagem da janela de encaixe kaggle / python: https://github.com/kaggle/docker-python

# Por exemplo, aqui estão vários pacotes úteis para carregar 



import numpy as np # álgebra Linear 

import pandas as pd # processamento de dados, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from datetime import date, datetime





# Arquivos de dados de entrada estão disponíveis no "../input/" directory.

# Por exemplo, executando isto (clicando em executar ou pressionando Shift+Enter) listará todos os arquivos no diretório de entrada



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Quaisquer resultados que você escrever no diretório atual serão salvos como saída.
df_op = pd.read_csv ('../input/OperacoesTesouroDireto.csv', sep='[;]')

df_ve = pd.read_csv ('../input/VendasTesouroDireto.csv', sep='[;]')

df_es = pd.read_csv ('../input/EstoqueTesouroDireto.csv', sep='[;]')
df_op.shape, df_ve.shape, df_es.shape
df_op.head(5)
df_op.info()
#Substituir "," por "." as colunas Quantidade, Valor do Titulo,Valor da Operacao

df_op['Quantidade']= df_op['Quantidade'].str.replace(',','.', regex=False)

df_op['Valor do Titulo']= df_op['Valor do Titulo'].str.replace(',','.', regex=False)

df_op['Valor da Operacao']= df_op['Valor da Operacao'].str.replace(',','.', regex=False)
df_op.tail(5).T
df_ve.info()
df_ve.head(5).T
#Substituir "," por "." as colunas Quantidade, Valor do Titulo,Valor da Operacao do dataframe df_ve

df_ve['PU']= df_ve['PU'].str.replace(',','.', regex=False)

df_ve['Quantidade']= df_ve['Quantidade'].str.replace(',','.', regex=False)

df_ve['Valor']= df_ve['Valor'].str.replace(',','.', regex=False)
df_ve.tail(5).T
df_es.info()
df_es.head(5).T
#Substituir "," por "." as colunas Quantidade, Valor do Titulo,Valor da Operacao do dataframe df_es

df_es['PU']= df_es['PU'].str.replace(',','.', regex=False)

df_es['Quantidade']= df_es['Quantidade'].str.replace(',','.', regex=False)

df_es['Valor Estoque']= df_es['Valor Estoque'].str.replace(',','.', regex=False)
df_es.tail(5).T
# Formatar data para ano-mes-dia

data = datetime.strptime('Data da Operacao', '%Y/%m/%d').date()

df_op['Data da Operacao'] = data.strftime('%Y/%m/%d')

df_op['Data da Operacao']
#Substituir "," por "."