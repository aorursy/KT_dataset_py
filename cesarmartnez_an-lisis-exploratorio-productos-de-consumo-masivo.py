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

#LIBRERIAS ADICIONALES

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px #Viz
productos_df = pd.read_excel('/kaggle/input/productos-consumo-masivo/output - Kaggle.xlsx', decimal=',')

productos_df 
#Tipo de variables

type(productos_df)

productos_df.dtypes
productos_df.info()
#Explorar datos de variable prod_unit_price

productos_df['prod_unit_price'].describe()
#eliminar columnas que no se utilizaran

productos_df = productos_df.drop('prod_icon',1)

productos_df = productos_df.drop('source_type',1)

productos_df = productos_df.drop('prod_units',1)

productos_df
# Group by Proveedor

proveedor_groups = productos_df.groupby("prod_source")

# Apply mean function to wieght column

count_proveedor = proveedor_groups['prod_unit_price'].mean()

count_proveedor

sns.barplot(count_proveedor, count_proveedor.index, palette="Greys")

plt.title("Precio promedio por producto de cada Proveedor", fontsize=20) # seting the title size

plt.xlabel("Promedio por producto", fontsize=12) # seting the 

plt.ylabel("Proveedor", fontsize=12) # seting the 
subcategory_count_df = productos_df.groupby(["subcategory"])["prod_unit_price"]. mean()

subcategory_count_df
sns.barplot(subcategory_count_df.values, subcategory_count_df.index, palette="icefire")

plt.title("Frecuencia Subcategorias vs precio promedio por producto", fontsize=20) # seting the title size

plt.xlabel("Promedio por producto", fontsize=12) # seting the 

fig=plt.gcf()

fig.set_size_inches(15,12)

plt.show()



a= productos_df[(productos_df['subcategory'] == 'Despensa')]

Despensa_df= a.sort_values("prod_unit_price", ascending=False)[:5]

Despensa_df

#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in a.prod_brand)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=3000, background_color="Black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.title("Marcas (Brand) mas frecuentes subcategoria Despensa", fontsize=24)

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
counts = a.prod_brand.value_counts()

most_frequent_category_names = counts[:10].index.tolist()

most_frequent_category_counts = counts[:10].values.tolist()





sns.barplot(most_frequent_category_names,most_frequent_category_counts,palette='icefire')

plt.title('Top 10 Marcas (Brand) con mayor Frecuencia de las subcategoria Belleza', fontsize=20)

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(15,9)

plt.show()

counts [:10]
#Top 5 productos de la Marca destacada MAGGI comparando su precio

maggi = productos_df[(productos_df['prod_brand'] == 'MAGGI')]

maggi_df = pd.DataFrame({'mean' : maggi.groupby(['prod_name']).prod_unit_price.max()}).reset_index()

maggi_df = maggi_df.sort_values(['mean'],ascending=False).reset_index(drop=True)

order_maggi = maggi['prod_name'][:5]

plt.figure(figsize = (12, 5))

plt.title('Top 5 productos de la marca MAGGI y sus respectivos precios unitarios',size=18)

sns.boxplot(x="prod_unit_price", y="prod_name", data=maggi, order= order_maggi ,palette='icefire')



# Different category distributions por tamaño

data = pd.DataFrame({'std' : a[a['date'] == 20190709].groupby(['subcategory', 'tags']).size()}).reset_index()

fig = px.treemap(data, path=['subcategory', 'tags'], values='std')

fig.show()

belleza= productos_df[(productos_df['subcategory'] == 'Belleza')]

belleza_df= belleza.sort_values("prod_unit_price", ascending=False)[:5]

belleza_df
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in belleza.prod_brand)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=3000, background_color="Black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.title("Marcas (Brand) mas frecuentes subcategoria Belleza", fontsize=24)

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
counts_belleza = belleza.prod_brand.value_counts()

most_frequent_category_names_belleza = counts_belleza[:10].index.tolist()

most_frequent_category_counts_belleza = counts_belleza[:10].values.tolist()





sns.barplot(most_frequent_category_names_belleza,most_frequent_category_counts_belleza,palette='icefire')

plt.title('Top 10 Marcas (Brand) con mayor Frecuencia de las subcategoria Belleza', fontsize=20)

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(15,8)

plt.show()
counts_belleza [:10]
#Top 5 productos de la Marca destacada VOGUE comparando su precio

vogue = productos_df[(productos_df['prod_brand'] == 'VOGUE')]

vogue_df = pd.DataFrame({'mean' : vogue.groupby(['prod_name']).prod_unit_price.max()}).reset_index()

vogue_df = vogue_df.sort_values(['mean'],ascending=False).reset_index(drop=True)

order_vogue = vogue['prod_name'][:5]

plt.figure(figsize = (10, 5))

plt.title('Top 5 productos de la marca VOGUE y sus respectivos precios unitarios',size=18)

sns.boxplot(x="prod_unit_price", y="prod_name", data=vogue, order= order_vogue ,palette='icefire')

# Different category distributions por tamaño

data_belleza = pd.DataFrame({'std' : belleza[belleza['date'] == 20190709].groupby(['subcategory', 'tags']).size()}).reset_index()

fig = px.treemap(data_belleza, path=['subcategory', 'tags'], values='std')

fig.show()

vinos= productos_df[(productos_df['subcategory'] == 'Vinos y Licores')]

vinos_df= vinos.sort_values("prod_unit_price", ascending=False)[:5]

vinos_df
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in vinos.prod_brand)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=3000, background_color="Black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.title("Marcas (Brand) mas frecuentes de la subcategoria Vinos y Licores", fontsize=24)

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
counts_vinos = vinos.prod_brand.value_counts()

most_frequent_category_names_vinos = counts_vinos[:10].index.tolist()

most_frequent_category_counts_vinos = counts_vinos[:10].values.tolist()





sns.barplot(most_frequent_category_names_vinos,most_frequent_category_counts_vinos,palette='icefire')

plt.title('Top 10 Marcas (Brand) con mayor Frecuencia de las subcategoria Vinos y Licores', fontsize=20)

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(20,10)

plt.show()
counts_vinos [:10]
#Top 5 productos de la Marca destacada NECTAR comparando su precio

nectar = productos_df[(productos_df['prod_brand'] == 'NECTAR')]

nectar_df = pd.DataFrame({'mean' : nectar.groupby(['prod_name']).prod_unit_price.max()}).reset_index()

nectar_df = nectar_df.sort_values(['mean'],ascending=False).reset_index(drop=True)

order_nectar = nectar['prod_name'][:5]

plt.figure(figsize = (12, 5))

plt.title('Top 5 productos de la marca NECTAR y sus respectivos precios unitarios',size=18)

sns.boxplot(x="prod_unit_price", y="prod_name", data=nectar, order= order_nectar ,palette='icefire')



# Different category distributions por tamaño

data_vinos = pd.DataFrame({'std' : vinos[vinos['date'] == 20190709].groupby(['subcategory', 'tags']).size()}).reset_index()

fig = px.treemap(data_vinos, path=['subcategory', 'tags'], values='std')

fig.show()



charcuteria= productos_df[(productos_df['subcategory'] == 'Charcutería')]

charcuteria_df= charcuteria.sort_values("prod_unit_price", ascending=False)[:5]

charcuteria_df
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in charcuteria.prod_brand)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=3000, background_color="Black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.title("Marcas (Brand) mas frecuentes subcategoria Vinos y Licores", fontsize=24)

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
counts_charcuteria = charcuteria.prod_brand.value_counts()

most_frequent_category_names_charcuteria = counts_charcuteria[:10].index.tolist()

most_frequent_category_counts_charcuteria = counts_charcuteria[:10].values.tolist()





sns.barplot(most_frequent_category_names_charcuteria,most_frequent_category_counts_charcuteria,palette='icefire')

plt.title('Top 10 Marcas (Brand) con mayor Frecuencia de las subcategoria Charcuteria', fontsize=20)

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(18,10)

plt.show()
counts_charcuteria [:10]
#Top 5 productos de la Marca destacada CENTURION FOODS comparando su precio

centurion = productos_df[(productos_df['prod_brand'] == 'CENTURION FOODS')]

centurion_df = pd.DataFrame({'mean' : nectar.groupby(['prod_name']).prod_unit_price.max()}).reset_index()

centurion_df = centurion_df.sort_values(['mean'],ascending=False).reset_index(drop=True)

order_centurion = centurion['prod_name'][:5]

plt.figure(figsize = (15, 5))

plt.title('Top 5 productos de la marca CENTURION FOODS y sus respectivos precios unitarios',size=18)

sns.boxplot(x="prod_unit_price", y="prod_name", data=centurion, order= order_centurion ,palette='icefire')
# Different category distributions por tamaño

data_charcuteria = pd.DataFrame({'std' : charcuteria[charcuteria['date'] == 20190709].groupby(['subcategory', 'tags']).size()}).reset_index()

fig = px.treemap(data_charcuteria, path=['subcategory', 'tags'], values='std')

fig.show()
