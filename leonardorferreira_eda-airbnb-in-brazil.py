# Carregando os pacotes do Python necessários à execução do Projeto de Data Science.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 18
sns.set_context('talk', font_scale=1.2);
%matplotlib inline
# Carregando o dataset com informações da plataforma Airbnb.

df = pd.read_csv("http://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2020-04-20/visualisations/listings.csv")
# Conhecendo o formato do dataset.

df.shape
# Conhecendo as varíaveis e seus tipos.

display(df.dtypes)
# Comando para visualizarmos as 5 primeiras linhas do Dataset.

df.head()

# Comando para visualizarmos as 5 últimas linhas do Dataset.

df.tail()
# Cálculo do percentual de dados faltantes no Dataset.

(df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
# Contrução dos histogramas das variáveis do dataset.

df.hist(bins=15, figsize=(15,15));
# Construindo os resumos estatísticos das variáveis numéricas usando a função describe.

df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365']].describe()
# Construção do Boxplot para variável minimum_nights.

df.minimum_nights.plot(kind='box', vert=False, figsize=(15, 3))
plt.show()

# Verificar a quantidade de valores acima de 30 dias para variável minimum_nights.

print("minimum_nights: valores acima de 30:")
print("{} entradas".format(len(df[df.minimum_nights > 30])))
print("{:.4f}%".format((len(df[df.minimum_nights > 30]) / df.shape[0])*100))
# Construção do Boxplot para variável price.

df.price.plot(kind='box', vert=False, figsize=(15, 3),)
plt.show()

# Verificar a quantidade de valores acima R$ 1500,00 para variável price.

print("\nprice: valores acima de 1500")
print("{} entradas".format(len(df[df.price > 1500])))
print("{:.4f}%".format((len(df[df.price > 1500]) / df.shape[0])*100))

# df.price.plot(kind='box', vert=False, xlim=(0,1300), figsize=(15,3));
# Buscou-se remover os outliers na construção de um novo dataset.

df_clean = df.copy()
df_clean.drop(df_clean[df_clean.price > 1500].index, axis=0, inplace=True)
df_clean.drop(df_clean[df_clean.minimum_nights > 30].index, axis=0, inplace=True)

# Buscou-se remover a variável `neighbourhood_group`, pois encontra-se vazia.

df_clean.drop('neighbourhood_group', axis=1, inplace=True)

# E na sequência buscou-se plotar novamente o histograma para as variáveis numéricas.

df_clean.hist(bins=15, figsize=(15,15));
# Construiu-se uma matriz de correlação entre as variáveis numéricas dataset df_clean.

corr = df_clean[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365']].corr()

display(corr)
# Na sequência buscou-se construir um mapa de calor (heatmap) a partir dessa matriz.

sns.heatmap(corr, cmap='RdBu', fmt='.3f', square=True, linecolor='white', annot=True);

# Identificando a quantidade de imóveis por tipo de imóvel disponível. Foi utilizado o método value_counts().

df_clean.room_type.value_counts()
# Verificando o percentual de cada tipo de imóvel no total do conjunto de dados.

df_clean.room_type.value_counts() / df_clean.shape[0]
# Construindo uma análise de uma varíavel (neighbourhood) em função de outra varíavel (price) usando a função groupby.

df_clean.groupby(['neighbourhood']).price.mean().sort_values(ascending=False)[:15]
# Verificando a quantidade de imóveis no bairro Vaz Lobo no contexto de todo dataset.

print(df_clean[df_clean.neighbourhood == "Vaz Lobo"].shape)

df_clean[df_clean.neighbourhood == "Vaz Lobo"]
# Verificando a quantidade de imóveis no bairro Engenheiro Leal no contexto de todo dataset.

print(df_clean[df_clean.neighbourhood == "Engenheiro Leal"].shape)

df_clean[df_clean.neighbourhood == "Engenheiro Leal"]
# Verificando a quantidade de imóveis no bairro Ricardo de Albuquerque no contexto de todo dataset.

print(df_clean[df_clean.neighbourhood == "Ricardo de Albuquerque"].shape)

df_clean[df_clean.neighbourhood == "Ricardo de Albuquerque"]
# Verificando a quantidade de imóveis no bairro Paciência no contexto de todo dataset.

print(df_clean[df_clean.neighbourhood == "Paciência"].shape)

df_clean[df_clean.neighbourhood == "Paciência"]
# Calculando as estatisticas descritivas do dataset df_clean com a retirada dos outliers.

df_clean[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365']].describe()