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
# Criando o DataFrame

df = pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv', sep='\t')



# Filtrando apenas dados relativos ao produto "GASOLINA COMUM"

df_gas = df[df['PRODUTO'] == 'GASOLINA COMUM']



# Excluindo colunas desnecessárias para análise

df_gas = df_gas.drop(columns=['Unnamed: 0', 'DATA INICIAL', 'DATA FINAL', 'PRODUTO', 'NÚMERO DE POSTOS PESQUISADOS',

                     'UNIDADE DE MEDIDA', 'DESVIO PADRÃO REVENDA', 'PREÇO MÍNIMO REVENDA', 'PREÇO MÁXIMO REVENDA',

                     'COEF DE VARIAÇÃO REVENDA', 'DESVIO PADRÃO DISTRIBUIÇÃO', 'PREÇO MÍNIMO DISTRIBUIÇÃO',

                     'PREÇO MÁXIMO DISTRIBUIÇÃO','COEF DE VARIAÇÃO DISTRIBUIÇÃO', 'MÊS'])



#Imprimindo um resumo para análise geral da situação dos dados.

df_gas.info()
# Ajustando os tipos de dados

df_gas['MARGEM MÉDIA REVENDA'] = df_gas['MARGEM MÉDIA REVENDA'].str.replace('-','0')

df_gas['MARGEM MÉDIA REVENDA'] = pd.to_numeric(df_gas['MARGEM MÉDIA REVENDA'])



df_gas['PREÇO MÉDIO DISTRIBUIÇÃO'] = df_gas['PREÇO MÉDIO DISTRIBUIÇÃO'].str.replace('-','0')

df_gas['PREÇO MÉDIO DISTRIBUIÇÃO'] = pd.to_numeric(df_gas['PREÇO MÉDIO DISTRIBUIÇÃO'])



# Criando coluna de percentual de lucro

perc_lucro = []



#Percorrer a coluna do dataframe e determinar as categorias

for valor in df_gas['PREÇO MÉDIO REVENDA']:

    perc_lucro = (df_gas['MARGEM MÉDIA REVENDA']/valor)

df_gas['perc_lucro'] = perc_lucro



df_gas.info()
# Importando bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns
# Realizando análise de distribuição

df_gas.drop(columns = ['ANO', 'perc_lucro']).describe()
# Gerando gráfico para análise de distribuição das variáveis

f, ax = plt.subplots(figsize=(15,6))

sns.distplot(df_gas['PREÇO MÉDIO REVENDA'], hist = True, kde = True, label='Preço Médio Revenda')

sns.distplot(df_gas['PREÇO MÉDIO DISTRIBUIÇÃO'], hist = True, kde = True, label='Preço Médio Distribuição')

sns.distplot(df_gas['MARGEM MÉDIA REVENDA'], hist = True, kde = True, label='Margem Média Revenda')



# Plot formatting

plt.legend(prop={'size': 12})

plt.title('Distribuição de Variáveis')

plt.xlabel('Valor em R$')

plt.ylabel('Frequência X 10%')  

# Gerando histograma de margem média por região

f, ax = plt.subplots(figsize=(15,6))

sns.boxplot(x="REGIÃO", y="perc_lucro", data=df_gas)
#Identificando os estados com maior margem de média na revenda.

df_gas.drop(columns = ['ANO']).groupby('ESTADO').mean().nlargest(5, 'MARGEM MÉDIA REVENDA')
# Explorando a correlação entre as variáveis



f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(df_gas.corr(), annot=True, fmt='.2f', linecolor='white', ax=ax, lw=.7)
#Gráfico de correlação 1

f, ax = plt.subplots(figsize=(15,6))

sns.regplot(data=df_gas,y='MARGEM MÉDIA REVENDA',x='PREÇO MÉDIO REVENDA', color='b')
#Gráfico de correlação 2

f, ax = plt.subplots(figsize=(15,6))

sns.regplot(data=df_gas,y='PREÇO MÉDIO REVENDA',x='ANO', color='b')
#Gráfico de correlação 3

f, ax = plt.subplots(figsize=(15,6))

sns.barplot(data=df_gas,y='perc_lucro',x='ANO', color='b')