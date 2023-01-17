import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
th_props = [

  ('font-size', '18px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#6d6d6d'),

  ('background-color', '#f7f7f9')

  ]



# Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '18px')

  ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]
df = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.shape
# Apresentando os tipos das colunas



df.dtypes
# Apresentando os 5 primeiros registros

df.head()
# Apresentando a quantidade de nulos por coluna

df.isnull().sum()
# Apresentando a quantidade de registros que possuem a coluna number_of_reviews igual a 0.



df[df["number_of_reviews"] == 0].count()
df.drop(['last_review'], axis = 1 , inplace = True )

df.drop(['host_name'], axis = 1 , inplace = True )

df.drop(['name'], axis = 1 , inplace = True )

df['reviews_per_month'].fillna("Unknown", inplace = True)

# criando tabela de quartis, max e min dos preços por bairro de NY

dados = []

for name, group in df[["neighbourhood_group", "price"]].groupby("neighbourhood_group"):

    group = group.describe(percentiles=[.25, .50, .75])

    group=group.iloc[1:]

    group.reset_index(inplace=True)

    group.rename(columns={'index':'Describe'}, inplace=True)

    group.rename(columns={'price':name}, inplace=True)

    dados.append(group)

        



dados=[df.set_index('Describe') for df in dados]

dados=dados[0].join(dados[1:])

(dados.style.set_table_styles(styles))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))



for ax in axes:

    ax.set_xticklabels([1,2,4,3,5], fontsize=16, rotation=30)





# criando grafico para demonstrar a média de preço por bairro

sns.barplot(x="neighbourhood_group", y="price", data = df.groupby("neighbourhood_group")["price"].mean().reset_index(), ax=axes[0]);



# criando grafico para demonstrar a quantidade de avaliações por bairro

sns.barplot(x="neighbourhood_group", y="number_of_reviews", data = df.groupby("neighbourhood_group")["number_of_reviews"].sum().reset_index(),  ax=axes[1]);
fig , ax  = plt.subplots(1,2 ,figsize = (20,5))



# Criando grafico de dispensão para demonstrar como está destribuido em um mapa os bairros de NY

sns.scatterplot(df.longitude, df.latitude , data = df , hue = "neighbourhood_group",ax = ax[0]);



# Criando grafico de dispensão para demonstrar como 

# está destribuido em um mapa os preços das acomodações de NY

sns.scatterplot(df.longitude,df.latitude,data = df[df['price'] < 500],hue = "price",ax = ax[1]);
fig , ax  = plt.subplots(1,2 ,figsize = (20,5))

sns.scatterplot(df.longitude, df.latitude , data = df , hue = "neighbourhood_group",ax = ax[0]);



sns.scatterplot(df.longitude, df.latitude , data = df , hue = "room_type",ax = ax[1]);
fig , ax  = plt.subplots(1,2 ,figsize = (20,5))

sns.barplot(x = df.room_type, y="price", data = df, hue = df.neighbourhood_group,ax = ax[0]);



sns.barplot(x = df.room_type, y="number_of_reviews", data = df, hue = df.neighbourhood_group,ax = ax[1]);