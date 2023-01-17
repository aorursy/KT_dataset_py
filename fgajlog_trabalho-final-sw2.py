import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style("whitegrid")

sns.set(font_scale=1.5)
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

df.head()
df.info()
#Checando valores nulos

df.isnull().sum()
#Checando se existe linhas duplicadas

df.duplicated().sum()

#Retirando colunas sem importância

df.drop(['name','id','host_name'], axis=1, inplace=True)

#Verificando as mudanças

df.head(5)

#Substituindo os nulos de "reviews_per_month" por 0

df.fillna({'reviews_per_month':0}, inplace=True)

#verificando...

df.reviews_per_month.isnull().sum()
#convertendo "last_review" para datetime

df["last_review"] = pd.to_datetime(df.last_review)

#Verificando as mudanças

df.info()
#preechendo "last_review"

df.last_review.fillna(method="ffill", inplace=True)

df.head()
#checando se acabou os nulos

df.isnull().sum()

df['neighbourhood_group'].value_counts()

#Manhattan é o mais antigo e mais densamente povoado dos cinco burgos que formam a cidade de Nova Iorque. 

#Contando o número de anúncios por bairro:

df['neighbourhood'].value_counts()

df['neighbourhood'].value_counts()

#cada burgo tem vários bairros, conforme abaixo:

df[['neighbourhood_group','neighbourhood']]
sns.set(font_scale=1.5)

sns.countplot(df['room_type'], palette="dark")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Tipo de instalação')
plt.figure(figsize=(12,8))

sns.scatterplot(df.longitude,df.latitude,hue=df.room_type, palette='dark')

plt.ioff()
sns.countplot(df['neighbourhood_group'], palette="dark")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Neighbourhood Group')
plt.figure(figsize=(12,8))

sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)

plt.ioff()
plt.figure(figsize=(12, 8))

plt.scatter(df.longitude, df.latitude, c=df.availability_365, cmap='winter', edgecolor='black', linewidth=0.8, alpha=0.75)



cbar = plt.colorbar()

cbar.set_label('availability_365')
plt.figure(figsize=(12, 8))

plt.scatter(df.longitude, df.latitude, c=df.price, cmap='summer', edgecolor='black', linewidth=0.8, alpha=0.75)



cbar = plt.colorbar()

cbar.set_label('Price $')

df[df.price == 0]

filter = df["price"]==0

  

# filtering data 

df_price0 = df.where(filter, inplace = False) 

  

# display 

df_price0.dropna(how='all', inplace=True)





plt.figure(figsize=(12, 8))

plt.scatter(df_price0.longitude, df_price0.latitude, c=df_price0.price, cmap='summer', edgecolor='black', linewidth=0.8, alpha=0.75)



cbar = plt.colorbar()

cbar.set_label('Price $')
pd.options.display.float_format = "{:.2f}".format

df.describe()
from wordcloud import WordCloud
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.neighbourhood))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('neighbourhood.png')

plt.show()