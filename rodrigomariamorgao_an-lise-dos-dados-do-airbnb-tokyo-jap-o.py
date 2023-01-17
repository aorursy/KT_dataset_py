# importar os pacotes necessarios

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# importar o arquivo listings.csv para um DataFrame

DATA_PATH = "../input/listings.csv"

df = pd.read_csv(DATA_PATH)
# mostrar as 5 primeiras entradas

df.head(5)
# identificar o volume de dados do DataFrame

print("Variáveis:\t {}\n".format(df.shape[1]))

print("Entradas:\t {}".format(df.shape[0]))



# verificar o tipo de entradas do dataset

display(df.dtypes)
# ordenar em ordem decrescente as variáveis por seus valores ausentes

(df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
# plotar histogramas

df.hist(bins=15, figsize=(15,10));
# calcular a média da coluna `price`

df.price.mean()
# criar uma matriz de correlação

corr = df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',

    'calculated_host_listings_count', 'availability_365']].corr()



# mostrar a matriz de correlação

display(corr)
# plotar um heatmap a partir das correlações

sns.heatmap(corr, cmap='RdBu', fmt='.2f', square=True, linecolor='white', annot=True);
# mostrar a quantidade de cada tipo de imóvel disponível

df.room_type.value_counts()
# mostrar a porcentagem de cada tipo de imóvel disponível

df.room_type.value_counts() / df.shape[0]
# ver preços por bairros, na média

df.groupby(['neighbourhood']).price.mean().sort_values(ascending=False)[:10]
# plotar os imóveis pela latitude-longitude

df.plot(kind="scatter", x='longitude', y='latitude', alpha=0.4, c=df['price'], s=8,

              cmap=plt.get_cmap('jet'), figsize=(12,8));

plt.ylim(35.4,35.9)
# ver a média da coluna `minimum_nights``

df.minimum_nights.mean()