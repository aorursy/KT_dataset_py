import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import matplotlib.pyplot as plt

from math import *
data=pd.read_csv("../input/googleplaystore.csv")

data.head()

data2=pd.read_csv("../input/googleplaystore_user_reviews.csv")

data2.head()
data.dropna()
data2.dropna()
data.drop_duplicates(inplace=True)
data['Rating'].describe()
plt.figure(figsize=(16,9), dpi= 60)

plt.hist(data['Rating'], 70, stacked=True, density=True,color="LightBlue")

plt.xlim(0, 5)

plt.title("Histograma de Rating", fontsize=22)

plt.ylabel("Frecuencia", fontsize= 18)

plt.xlabel("Rating",fontsize= 18)

plt.show()

data2['Sentiment'].describe()

plt.figure(figsize=(16,10), dpi= 70)

g1 = sns.countplot(x="Sentiment",data=data2, palette = "Set1")

g1

plt.title('Percepción de las Apps',size = 20)

plt.xlabel("Sentimiento", size=18)

plt.ylabel("Frecuencia", size=18)
plt.figure(figsize=(16,9), dpi= 80)

g2 = sns.countplot(x="Category",data=data, palette = "Set2")

g2.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g2 

plt.title('Conteo de apps descargadas por Categoria',size = 22)

plt.xlabel("Categoria", size=18)

plt.ylabel("Frecuencia", size=18)

plt.show()
plt.figure(figsize=(16,9), dpi= 60)

g3 = sns.countplot(x="Type",data=data, palette = "Set2")

g3 

plt.title('Conteo de apps descargadas por Tipo',size = 20)

plt.xlabel("Tipo", size=16)

plt.ylabel("Frecuencia", size=16)

plt.show()
data.sort_values(by='Reviews',ascending=False)[:10]
data.sort_values(by='Installs', ascending=False)[:10]
