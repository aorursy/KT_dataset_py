# Imports and Basic Data Cleaning (Drop Duplicates and NaN data)
import numpy as np
import pandas as pd 
import os
from matplotlib import pyplot as plt
import seaborn as sns
!pip install googletrans
plt.style.use('fivethirtyeight')

df = pd.read_csv('/kaggle/input/mercadona-es-product-pricing/thegurus-opendata-mercadona-es-products.csv')
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

df.head()
df.info()
print('Metrics for Price:')
df.price.describe()
print('Metrics for Reference Price:')
df.reference_price.describe()
from googletrans import Translator
translator = Translator()

df = df.replace('_', ' ',regex=True)

category_unique = list(df.category.unique())
name_unique = df.name.unique()
category_translations = translator.translate(category_unique, src='es', dest='en')

def translate_category(cat_text):
    index = category_unique.index(cat_text)
    return category_translations[index].text
    
df['category_english'] = df['category'].apply(lambda x: translate_category(x))
    

sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15,5))
g = sns.countplot(x='category_english',data=df, order=df.category_english.value_counts().iloc[:50].index)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.fig.set_size_inches(20, 10)
plt.title('Top 50 Product Categories')
plt.ylabel('Number of Products')
plt.xlabel('Product Category')
plt.show()

plt.figure(figsize=(15,5))
g = sns.countplot(x='category',data=df, order=df.category.value_counts().iloc[:50].index)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.fig.set_size_inches(20, 10)
plt.title('50 Categorías de Productos más Grandes')
plt.ylabel('Número de Producto')
plt.xlabel('Categoria de Producto')
plt.show()



g = sns.catplot(x="price", y="category_english", kind="box", data=df,showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, order=df.category_english.value_counts().iloc[:50].index)
g.set(xlim=(0, 25))
g.fig.set_size_inches(40, 20)
plt.xlabel('Price (€)')
plt.ylabel('Product Category')
plt.title("Price by Product Category - Ordered by Category Frequency")
plt.show()

g = sns.catplot(x="price", y="category", kind="box", data=df,showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, order=df.category.value_counts().iloc[:50].index)
g.set(xlim=(0, 25))
g.fig.set_size_inches(40, 20)
plt.xlabel('Precio (€)')
plt.ylabel('Categoria de Producto')
plt.title("Precio por Categoría de Producto - Ordenado por Frecuencia de Categoría")
plt.show()



df_cat = df.groupby("category_english").median().sort_values(by = 'price', ascending=False)
g = sns.catplot(x="price", y="category_english", kind="box",showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, order=df_cat.iloc[:50].index)
g.set(xlim=(0, 25))
g.fig.set_size_inches(40, 20)
plt.xlabel('Price (€)')
plt.ylabel('Product Category')
plt.title("Price by Product Category - Ordered by Median Price")
plt.show()

df_cat = df.groupby("category").median().sort_values(by = 'price', ascending=False)
g = sns.catplot(x="price", y="category", kind="box",showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, order=df_cat.iloc[:50].index)
g.set(xlim=(0, 25))
g.fig.set_size_inches(40, 20)
plt.xlabel('Precio (€)')
plt.ylabel('Categoria de Producto')
plt.title("Precio por Categoría de Producto - Ordenado por Precio Medio")
plt.show()



df_cat = df.groupby("category_english").mean().sort_values(by = 'price', ascending=False)
g = sns.catplot(x="price", y="category_english", kind="box",showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, order=df_cat.iloc[:50].index)
g.set(xlim=(0, 45))
g.fig.set_size_inches(40, 20)
plt.xlabel('Price (€)')
plt.ylabel('Product Category')
plt.title("Price by Product Category - Ordered by Mean Price")
plt.show()

df_cat = df.groupby("category").mean().sort_values(by = 'price', ascending=False)
g = sns.catplot(x="price", y="category", kind="box",showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, order=df_cat.iloc[:50].index)
g.set(xlim=(0, 45))
g.fig.set_size_inches(40, 20)
plt.xlabel('Precio (€)')
plt.ylabel('Categoria de Producto')
plt.title("Precio por Categoría de Producto - Ordenado por Precio Promedio")
plt.show()