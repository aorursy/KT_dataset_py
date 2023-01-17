# Antes de qualquer coisa, vamos importar a biblioteca Pandas

import pandas as pd
# Vamos criar um DataFrame do resultado de uma votação.

pd.DataFrame({'Sim':[10,15], 'Nao':[21, 4]})
# Uma Serie tem apenas uma lista. Vamos criar uma Serie:

pd.Series([10,15])
# 1.1 Crie um DataFrame que tenha 3 colunas e 2 duas linhas com o nome: vendas

# 1.2 Crie uma Serie que contenha o total dos produtos. Com os seguintes valores: 600, 40

# 2. Salve o dataframe vendas criado no exercício 1.1 num arquivo csv chamado `vendas`

# Lista os arquivos no diretório

import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Omite os warnings.. por enquanto

import warnings

warnings.filterwarnings('ignore')



# Importa o pandas

import pandas as pd



# Vamos abrir o arquivo em Excel (.xlsx) de uasgs

uasgs = pd.ExcelFile('/kaggle/input/uasgs.xlsx')



# Transforma de Excel para DataFrame. O Parâmetro aqui é o nome da aba (Sheet 1)

uasgs = uasgs.parse('itensUASGago2019') 



# Visualiza as primeiras linhas do dataset 

uasgs.head()
# DICA ESPECIAL: Mostrar mais colunas, opção do Pandas para não aparecer o "..." 

pd.set_option('display.max_columns', 500)

uasgs.head()
# Importa o arquivo CSV de licitação com o nome de licit

licit = pd.read_csv('../input/licitacao.csv')



# Visualiza as primeiras linhas do dataset 

licit.head()
# 3. Faça a importação do arquivo CSV de modalidade e visualize suas primeiras linhas.

# 4.1 Renomeie a coluna codmodalidade do dataset modalidade para codigomodalidade

# 4.2 Faça a combinação para juntar os datasets de licitações e modalidades e salve num novo dataset chamado licitm

licit.head()
# EXEMPLOS



# Retorna apenas a coluna dataPublicacao 

licit['dataPublicacao']



# Retorna apenas a segunda data

licit.dataPublicacao[1]

licit.dataPublicacao.iloc[1]



# Retorna as primeiras 5 linhas

licit[:5]



# Retorna apenas a ultima coluna

licit.iloc[:,-1]



# Cria um dataset de exemplo apenas com as linhas 3 e 5 das colunas dataPublicacao e situacao

exemplo = licit.loc[[3,5],['dataPublicacao','situacao']]

exemplo



# Retorna um dataset contendo apenas publicacoes do ano de 2019. Dataset com o nome de: a2019

a2019 = licit[licit['dataPublicacao'].str.contains('2019')]

a2019.head()
# 5.1. Crie um dataset apenas com as colunas dataPublicacao e situacao de todos do ano de 2018: a2018

# Importa datetime

import datetime



# Imprime a data e hora

print(datetime.datetime.now())



# Imprime apenas a data

print(datetime.date.today())



# Imprime apenas o ano

print(datetime.date.today().year)



# Imprime apenas o mês

print(datetime.date.today().month)



# Imprime apenas o dia

print(datetime.date.today().day)



# Imprime apenas a hora

print(datetime.datetime.today().hour)
# Outro exemplo para imprimir o ano

agora = datetime.datetime.now()

agora.strftime('%Y')
# Convertendo o formato da data de 2019-08-26 para 26/08/2019

datetime.datetime.strptime("2019-08-26", '%Y-%m-%d').strftime('%d/%m/%Y')
# Cria uma nova coluna com a data formatada de 26/08/2019 para 26-08-2019

licit['dataFormatada'] = [datetime.datetime.strptime(x, '%d/%m/%Y').strftime('%d-%m-%Y') for x in licit['dataPublicacao']]
# 1. Crie uma nova coluna no dataset de licit para o ano de publicacao: ano

# 1. Faça um agrupamento para descobrir a quantidade de licitacoes publicadas por ano

# 2. Crie um DataFrame com o resultado da quantidade de licitacoes por ano

# 3. Ordene os valores e responda: Qual foi o ano que teve o menor número de licitacoes publicadas?
# Verifica se tem algum valor NaN no Dataset inteiro

licit.isnull().sum()



# Verifica a quantidade de valores NaN apenas em uma coluna

licit['dataPublicacao'].isnull().sum()
# 1. Importa o arquivo CSV de itens licitação com o nome de item



# 2. Verifica se tem algum valor NaN



# 3 Apague os registros NaN de licit. Quantas linhas restou no dataset?

# 1.1 RESPOSTA

# 1.1 Crie um DataFrame que tenha 3 colunas e 2 duas linhas com o nome: vendas

# Colunas:             Produto | Quantidade | Preço

# Na primeira linha: Chocolate |    200     | 3,00 

# Na segunda linha:     Banana |     80     | 0,50

vendas = pd.DataFrame({'Produto':['Chocolate', 'Banana'], 'Quantidade':[200, 80], 'Preço':[3.00, 0.50]})

vendas.head()
# 1.2 RESPOSTA

# Crie uma Serie que contenha o total dos produtos. Com os seguintes valores: 600, 40

pd.Series([600, 40])
# RESPOSTA 2. Salve o dataframe vendas criado no exercício 1.1 num arquivo csv chamado `vendas`. 

vendas.to_csv('vendas.csv')
# 3. Faça a importação do arquivo CSV de modalidade e visualize suas primeiras linhas.

import pandas as pd

modali = pd.read_csv('../input/modalidade.csv')

modali.head()
# 4.1 Renomeie a coluna codmodalidade do dataset modalidade para codigomodalidade

modalidade = modalidade.rename(columns={'codmodalidade':'codigomodalidade'})

modalidade.head()
# 4.2 Faça a combinação para juntar os datasets de licitações e modalidades e salve num novo dataset chamado licitm

licitm = pd.merge(licit, modali, on='modalidade')

licitm.head()



# Pode ser feito Usando o join também

licitm2 = licit.set_index('modalidade').join(modalidade.set_index('modalidade'))

licitm2.head()
# 5.1. Crie um dataset apenas com as colunas dataPublicacao e situacao de todos do ano de 2018: a2018

a2018 = licitm.loc[licitm['dataPublicacao'].str.contains('2018'),['dataPublicacao','situacao']]
# 6.1. Crie uma nova coluna no dataset de licit para o ano de publicacao: ano

licit['ano'] = [datetime.datetime.strptime(x, '%d/%m/%Y').strftime('%Y') for x in licit['dataPublicacao']]

licit.head()
# 7.1 Faça um agrupamento para descobrir a quantidade de licitacoes publicadas por ano

licit.groupby('ano').numLicitacao.count()   # Cria uma Serie

licit.groupby('ano')[['numLicitacao']].count()  # Cria um DataFrame



# 7.2 Crie um DataFrame com o resultado da quantidade de licitacoes por ano

licitporano = licit.groupby('ano')[['numLicitacao']].count()



# 7.3 Ordene os valores e responda: Qual foi o ano que teve o menor número de licitacoes publicadas?

licitporano.sort_values(by='numLicitacao', ascending=False)
# 8.1. Importa o arquivo CSV de itens licitação com o nome de item

item = pd.read_csv('../input/itemLicitacao.csv')

item.head()



# 8.2. Verifica se tem algum valor NaN

item.isnull().sum()



# 8.2. ou

len(item[item.codigoOrgao.isnull()])



# 8.3 Apague os registros NaN de licit. Quantas linhas restou no dataset?

licit.dropna(inplace=True)

licit.shape