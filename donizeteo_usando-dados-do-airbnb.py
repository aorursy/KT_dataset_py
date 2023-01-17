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
# Criando o dataframe



nyc19 = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# Verificando o tamanho do Dataframe

print('Dados AIBNB Nova Work: ', nyc19.shape)
# Verificando informações sobre o dataframe

nyc19.info()
# listar alguns dados

nyc19.head(5)
# Alterando nome de algumas colunas:



nyc19 = nyc19.rename(columns={'name': 'Nome','host_name': 'Nome do Anfitrião' ,'neighbourhood_group': 'Grupo Bairro','neighbourhood':'Vizinhança','room_type':'Tipo de sala', 'price' : 'preço',

                     'minimum_nights':'Noites mínimas', 'number_of_reviews':'Número de comentários', 'last_review':'Última revisão', 'reviews_per_month':'Comentários por mês',

                     'calculated_host_listings_count':'Total Listagens de Hosts', 'availability_365':'Disponibilidade'  })
# mostransdo os 5 últimos registros

nyc19.tail()
# listar alguns dados

nyc19.head(6).T
# Descrevendo dados

# Podemos observar que a média dos comentários é de 1,37 ao mês e  no máximo 58,50 no mês, bem como hospedagem com disponibilidade mínima igual a 0 (zero) e máxima de 365 dias (ano). 

# Em média temos 112,7 dias de disponibilidade no ano.

nyc19.describe()
# Verificando novamente informações sobre o dataframe

nyc19.info()
#Pesquisando sobre as hospedagens que não tem nome:

nyc19[nyc19['Nome'].isnull()]
# TOTAL de hospedagens que não tem nome:.isnull().value_counts()]

S_Nome = nyc19[nyc19['Nome'].isnull()]

#Podemos observar abaixo que 16 estabelecimentos não definiram um nome para sua hospedagem.

S_Nome.info()
# Verificando novamente informações sobre os dados no dataframe

nyc19.head()
# Verificando se há duplicidade de dados na base

nyc19[nyc19.duplicated(keep=False)]
# Verificando os imóveis com preços de "400"



nyc19.loc[(nyc19['preço']==400)]
# Gerar somatório por grupo/bairro



nyc19["Grupo Bairro"].value_counts()
#usando o maplotlib para geração de gráficos

import matplotlib.pyplot as plt



# Gerando grafico por Bairro

nyc19["Grupo Bairro"].value_counts().plot.bar()
#Avaliando dados sobre "Vizinhança"

nyc19["Vizinhança"].value_counts()
# O gráfico abaixo apresenta muitas informações sobre Vizinhança. Mantive com objetivo de avaluiar outras opções

plt.figure(figsize=(20,5))

nyc19["Vizinhança"].value_counts().plot.bar()

plt.xticks(rotation=65)

plt.show()
# Gerando dados e grafico por Vizinhança



plt.figure(figsize=(20,5))

nyc19["Vizinhança"].value_counts().plot()

plt.xticks(rotation=65)

plt.show()
# Ordenando pelo Grupo Bairro e preço das diárias:



nyc19_ = nyc19.sort_values(['Grupo Bairro', 'preço'])

nyc19_.head(7).T

#Outra presquisa ordenando por preço e grupo Bairro e apesentando o resultado os últimos 30 registros.

nyc19_ = nyc19.sort_values(['preço', 'Grupo Bairro'])

nyc19_[-30:]

# Valor total se todas as hospedagem estivessem reservadas

nyc19_.groupby('Grupo Bairro')['preço'].sum().sort_values(ascending=False)
#Usando o seaborn para elaborar gráficos

import seaborn as sns

import matplotlib.pyplot as plt



# Gráfico do valor total se todas as hospedagem estivessem reservadas

plt.figure(figsize=(10,5))

sns.barplot(x='Grupo Bairro', y='preço', data=nyc19_)

plt.show()
# Gráfico do valor total se todas as hospedagem estivessem reservadas (usando gráfico cde linhas)

plt.figure(figsize=(10,5))

sns.pointplot(x='Grupo Bairro', y='preço', data=nyc19_)

plt.title('Valor total diarias hospedagem')

plt.grid(True, color='grey')
