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
# Carregando os dados do desmatamento do INPE

desm_amazon_inpe = pd.read_csv('/kaggle/input/desm-amazon-inpe/desm_amazon_inpe.csv', sep= ',', encoding='ISO-8859-1')

                               

desm_amazon_inpe.head()                               
desm_amazon_inpe.info()
# Obtendo todos os registros duplicados do INPE

desm_amazon_inpe[desm_amazon_inpe.duplicated(keep=False)]
# Eliminando os registros repetidos

desm_amazon_inpe.drop_duplicates(inplace=True)

desm_amazon_inpe.info()
# Carregando os dados de produção do IBGE

prod_amazon_ibge = pd.read_csv('/kaggle/input/prod-amazon-ibge/prod_amazon_ibge.csv', sep= ',', encoding='ISO-8859-1')



prod_amazon_ibge.head()
# Verificando a existência de registros duplicados nos dados do IBGE

prod_amazon_ibge[prod_amazon_ibge.duplicated(keep=False)]
# Verificando os tamanhos dos dataframes

print('Desmatamento:', desm_amazon_inpe.shape)

print('Produção:', prod_amazon_ibge.shape)
# Juntando os dataframe

# Codigo de exemplo result = pd.merge(left, right, on='key')

dados_2018 = pd.merge(desm_amazon_inpe,prod_amazon_ibge,  how='inner', on=['codigo','municipio'])



dados_2018.info()
# Para que os dados estejam padronizados e com mesma medida tanto no INPE como no IBJE,

# trasformamos área desmatada em km² da série histórica para hectares.

columns = dados_2018.columns[2:23]

dados_2018[columns] *= 100



dados_2018['DESM'] = dados_2018['area_1ha_2018']



dados_2018.head()
# Seleciona as variáveis que compoem a serie hístórica do desmatamento na Amazônia Legal nos ultimos 11 ansos

dados_2018_0 = dados_2018[['area_1ha_2008','area_1ha_2009','area_1ha_2010','area_1ha_2011','area_1ha_2012','area_1ha_2013','area_1ha_2014','area_1ha_2015','area_1ha_2016','area_1ha_2017','area_1ha_2018']]



dados_2018_0.head(10)                   
# Calcula o total de desmatamento por ano.

a = dados_2018_0.T.sum(axis=1).reset_index()

a['index'] = a['index'].str[-4:] 

a = a.rename(columns={'index': 'Ano', 0: 'DESM'})

a
# Importando a biblioteca gráfica matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
# Gráfico de linhas dos incrementos de desamtametno na Amazonia Legal nos ultimos onze anos

plt.figure(figsize=(15,5))

sns.pointplot(x='Ano', y='DESM', data=a, color='green')

plt.title('Incrementos de desamtametno na Amazonia Legal nos ultimos onze anos')

plt.grid(True, color='grey')

plt.ylabel('Desmatamento (ha)')

plt.show()
# Estatísticas descritivas para a serie histórica do desmatamento na Amazônia Legal

dados_2018_0.describe()
# Selecão das 10 variaveis determinantes na composição do modelo do desmatamento na Amazõnia Legal em 2018

dados_2018_1 = dados_2018[['codigo','municipio','estado','DESM','BOV','LAVTEMP','LAVPERM','MAD','PIB_MUN','POP','SOJA','MILHO','MANDIOC','ARROZ']]



dados_2018_1.head()
# Deleta as linhas cujas valores são missing para a variável desmatamento(DESM)

# Só iremos estudar os munícipios que tiveram dados sobre o desmatamento

dados_2018_1.dropna(thresh=14)
# Os cinco municípios que mais desmataram na Amazonia Legal em 2018



dados_2018_1.nlargest(5,'DESM')

# Os cinco municípios que mais desmataram na Amazonia Legal em 2018

plt.figure(figsize=(10,5))

plt.title('Os cinco municípios que mais desmataram na Amazonia Legal em 2018')

sns.barplot(x='DESM', y='municipio',data=dados_2018_1.nlargest(5,'DESM')[:80])

plt.xlabel('Desmatamento (ha)')

plt.show()
# Agrupando o dataframe do total de desamtamento por estado

total_desm_2018  = dados_2018_1.groupby('estado')['DESM'].sum().reset_index()

# Total de área desmatada por estado da Amazônia Legal em 2018

total_desm_2018  = total_desm_2018.sort_values(by='DESM', ascending=False)



#plotanto o grafico de barras

plt.figure(figsize=(10,5))

plt.title('Total de área desmatada por estado da Amazônia Legal em 2018')

sns.barplot(x='estado', y='DESM', data=total_desm_2018[:80])

plt.xlabel('Estado')

plt.ylabel('Desmatamento (ha)')

plt.show()
# Correlação entre as 10 variáveis determinantes para o desmatamento da Amazônia.



dados_2018_2 = dados_2018_1[['DESM','BOV','LAVTEMP','LAVPERM','MAD','PIB_MUN','POP','SOJA','MILHO','MANDIOC','ARROZ']]

dados_2018_2.corr()



# Plotando a correlação

# Aumentando a area do gráfico

f, ax =plt.subplots(figsize=(10,5))

sns.heatmap(dados_2018_2.corr(), annot=True, fmt='.2f', linecolor='black', lw=.7, ax=ax)

plt.title('Correlação entre as 10 variáveis determinantes para o desmatamento da Amazônia Legal')

plt.show()
# <h3>1.5 - Gráfico de dispersão entre as variávis mais correlacionadas e o desmantamento da Amazônia Legal em 2018</h3>

plt.figure(figsize=(10,5))

sns.scatterplot(dados_2018_1['DESM'], 

               dados_2018_1['BOV'], color='red')

plt.title('Gráfico de dispersão entre bovinos e o desmantamento em 2018')

plt.xticks(rotation=0)

plt.locator_params(axis='y', nbins=20)

plt.xlabel('Desmatamento (ha)')

plt.ylabel('Bovinos (nº cabeças)')

plt.show()
# cálculo do percentual de área desmatada

# com base nos quartis da area_1ha_2018



q3 =dados_2018_1['DESM'].quantile(0.75)

q2 =dados_2018_1['DESM'].quantile(0.5)

q1 =dados_2018_1['DESM'].quantile(0.25)



Percentual_Desmatamento = []



# Percorrer a coluna do dataframe e determinar as categorias

for valor in dados_2018_1['DESM']:

    if valor >= q3 :

        Percentual_Desmatamento.append('> 75')

    elif valor < q3 and valor >= q2:

        Percentual_Desmatamento.append('50-75')

    elif valor < q2 and valor >= q1:

        Percentual_Desmatamento.append('25-50')

    else:

        Percentual_Desmatamento.append('< 25')

        

dados_2018_1['Percentual_Desmatamento'] = Percentual_Desmatamento

        

dados_2018_1.head()
# Boxplot percentual de área desmatada por faixa e numero de cabeças de bovinos

plt.figure(figsize=(10,5))

sns.boxplot(dados_2018_1['Percentual_Desmatamento'], dados_2018_1['BOV'])

plt.title('Boxplot do percentual de área desmatada por faixa e numero de cabeças de bovinos')

plt.xticks(rotation=65)

plt.locator_params(axis='y', nbins=20)

plt.ylabel('Bovinos (nº cabeças)')

plt.xlabel('Percentual de área desmatada por faixa')

plt.show()
# Gráfico de dispersão entre bovinos e o desmantamento da Amazônia Legal por faixas de classe (percentual de área desmatada).



plt.figure(figsize=(10,5))

sns.scatterplot(dados_2018_1['DESM'], 

               dados_2018_1['BOV'],

               hue=dados_2018_1['Percentual_Desmatamento'],

               style=dados_2018_1['Percentual_Desmatamento'])

plt.title('Gráfico de dispersão entre bovinos e o desmantamento da Amazônia Legal por faixas de classe')

plt.xticks(rotation=0)

plt.locator_params(axis='y', nbins=20)

plt.xlabel('Desmatamento (ha)')

plt.ylabel('Bovinos (nº cabeças)')

plt.show()
# Boxplot do desmatamento da Amazônia Legal em 2018

plt.figure(figsize=(12,5))

sns.boxplot(dados_2018_1['DESM'])

plt.title('Boxplot do desmatamento da Amazônia Legal em 2018')

plt.xticks(rotation=0)

plt.locator_params(axis='y', nbins=20)

plt.xlabel('Desmatamento (ha)')

plt.show()
# Histograma do desmatamento da Amazônia Legal em 2018

dados_2018_1.hist(column='DESM', bins=25, grid=False, figsize=(10,5), color='#86bf91', zorder=2, rwidth=0.9)

plt.title('Histograma do desmatamento da Amazônia Legal em 2018')

plt.xlabel('Desmatamento (ha)')

plt.show()