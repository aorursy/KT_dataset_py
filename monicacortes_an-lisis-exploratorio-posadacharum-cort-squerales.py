#Helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Viz

import seaborn as sns #Viz

import plotly.express as px #Viz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/productos-consumo-masivo/output - Kaggle.xlsx', decimal=',')

# Print the head of df

df.head(3)
# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)
# Statistics for continuous variables

df.describe()
# Statistics for categorical variables

print(pd.DataFrame(df['date'].value_counts(dropna=False)))

df.describe(include=[np.object])
#información de los datos

print('Total de registros:',df.shape[0])

print('Número de fuente de productos:',df['prod_source'].nunique())

print(df['subcategory'].value_counts())

print

print('Subcategoria con mayor número de productos.:',df['subcategory'].value_counts().index[0],'con',df['subcategory'].value_counts().values[0],'productos')
#Productos por subcategoria



plt.subplots(figsize=(22,15))

sns.countplot(y=df['subcategory'],order=df['subcategory'].value_counts().index)

plt.show()

#Información de las marcas.

print('Número de marcas:',df['prod_brand'].nunique())

print(df['prod_brand'].value_counts())


#Información marcas top 15 

datos_marca=df['prod_brand'].value_counts()[:15].to_frame()

sns.barplot(datos_marca['prod_brand'],datos_marca.index,palette='inferno')

plt.title('Top 15 de marcas')

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()

# Subcategoria belleza  respecto al precio por unidad

databelleza = df[df['subcategory'] == 'Belleza']

plt.title('Subcategoria belleza',size=15)

sns.boxplot(x="prod_unit_price", y="subcategory", data=databelleza,palette='inferno')
# Distribution by date and supermarket

databelleza = df[df['subcategory'] == 'Belleza']

data = databelleza.groupby(['date', 'prod_source']).size()

sns.barplot(data.values, data.index, palette='inferno')

plt.title('Distribución por fecha y supermecado de la subcategoria belleza')
#tags categoria belleza

dataTagsBelleza = pd.DataFrame({'count' : df[df['subcategory'] == 'Belleza'].groupby(['tags']).size()}).reset_index()

fig = px.treemap(dataTagsBelleza, path=[ 'tags'], values='count',  title='Tags de la subcategoria belleza')



fig.show()
# Distribution by tags of subcategory belleza

databelleza = df[df['subcategory'] == 'Belleza']

data = databelleza.groupby(['tags']).size()

sns.barplot(data.values, data.index, palette='inferno')

plt.title('Distribución tags de belleza')


#marca subcategoria belleza

dataTagsBelleza = pd.DataFrame({'count' : df[df['subcategory'] == 'Belleza'].groupby(['prod_brand']).size()}).reset_index()

fig = px.treemap(dataTagsBelleza, path=[ 'prod_brand'], values='count',  title='Marcas de la subcategoria belleza')



fig.show()
#Marcas de la subcategoria belleza

databelleza = df[df['subcategory'] == 'Belleza']

databellezaM=databelleza['prod_brand'].value_counts()[:15].to_frame()

sns.barplot(databellezaM['prod_brand'],databellezaM.index,palette='inferno')

plt.title('Top 15 de marcas de la subcategoria belleza')

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()



databellezaM.head(15)

# For all the dates marcas de la subcategoria Belleza

plt.figure(figsize = (10, 20))

plt.subplots_adjust(hspace=0.1, wspace=1)

pal = sns.color_palette("inferno", 20)



databelleza = df[df['subcategory'] == 'Belleza']

i = 1

for date in databelleza['date'].unique():

    data = databelleza[databelleza['date'] == date].groupby(['prod_brand']).size()      

    plt.subplot(2, 2, i)

    sns.barplot(data.values, data.index, palette=pal)

    i = i + 1


#Productos de la marca vogue vs precio 

datosVogue = df[(df['prod_brand'] == 'VOGUE')]

data1 = pd.DataFrame({'mean' : df[(df['prod_brand'] == 'VOGUE')].groupby(['prod_name']).prod_unit_price.mean()}).reset_index()

data1 = data1.sort_values(['mean'],ascending=False).reset_index(drop=True)

order = data1['prod_name'][:10]



data2 = df[(df['prod_brand'] == 'VOGUE')]

plt.figure(figsize = (5, 10))

plt.title('Productos con sus precios de la marca Vogue',size=15)

sns.boxplot(x="prod_unit_price", y="prod_name", data=datosVogue, order= order ,palette='inferno')



print('Número de productos vogue:',datosVogue['prod_name'].nunique())



data1.head(10)
#Productos de la marca NUTRISSE vs precio 

datosNUTRISSE = df[(df['prod_brand'] == 'NUTRISSE')]

data1 = pd.DataFrame({'mean' : df[(df['prod_brand'] == 'NUTRISSE')].groupby(['prod_name']).prod_unit_price.mean()}).reset_index()

data1 = data1.sort_values(['mean'],ascending=False).reset_index(drop=True)

order = data1['prod_name'][:5]



data2 = df[(df['prod_brand'] == 'NUTRISSE')]

plt.figure(figsize = (5, 10))

plt.title('Productos con sus precios de la marca NUTRISSE',size=15)

sns.boxplot(x="prod_unit_price", y="prod_name", data=datosNUTRISSE, order= order ,palette='inferno')



print('Número de productos NUTRISSE:',datosNUTRISSE['prod_name'].nunique())



data1.head(16)