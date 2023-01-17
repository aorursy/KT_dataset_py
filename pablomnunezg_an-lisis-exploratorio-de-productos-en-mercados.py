!pip install plotly==4.5.2
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
# NaN values
df.isnull().sum()

# Print the info of df
print(df.info())

# Print the shape of df
print(df.shape)
# Statistics for continuous variables
df.describe()
# Statistics for categorical variables
print(pd.DataFrame(df['date'].value_counts(dropna=False)))
df.describe(include=[np.object])
# Distribution by category and subcategory 
data = df.groupby(['category', 'subcategory']).size()
sns.barplot(data.values, data.index, palette="husl")
#Información de las marcas.
print('Número marcas:',df['prod_brand'].nunique())
print(df['prod_brand'].value_counts())
data = pd.DataFrame({'count' : df[df['date'] == 20191101].groupby(['subcategory', 'prod_brand']).size()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='count')
fig.update_layout(title_text='Map Count brand 20191101 ')
fig.show()

data = pd.DataFrame({'mean' : df[df['date'] == 20191101].groupby(['subcategory', 'prod_brand']).prod_unit_price.mean()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='mean')
fig.update_layout(title_text='Map mean prices  20191101')
fig.show()


data = pd.DataFrame({'count' : df[df['date'] == 20200220].groupby(['subcategory', 'prod_brand']).size()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='count')
fig.update_layout(title_text='Map Count brand 20200220 ')
fig.show()

data = pd.DataFrame({'mean' : df[df['date'] == 20200220].groupby(['subcategory', 'prod_brand']).prod_unit_price.mean()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='mean')
fig.update_layout(title_text='Map mean prices  20200220')
fig.show()
data = pd.DataFrame({'count' : df[df['date'] == 20190709].groupby(['subcategory', 'prod_brand']).size()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='count')
fig.update_layout(title_text='Map Count brand 20190709 ')
fig.show()

data = pd.DataFrame({'mean' : df[df['date'] == 20190709].groupby(['subcategory', 'prod_brand']).prod_unit_price.mean()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='mean')
fig.update_layout(title_text='Map mean prices  20190709')
fig.show()
data = pd.DataFrame({'count' : df[df['date'] == 20190609].groupby(['subcategory', 'prod_brand']).size()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='count')
fig.update_layout(title_text='Map Count brand 20190609 ')
fig.show()

data = pd.DataFrame({'mean' : df[df['date'] == 20190609].groupby(['subcategory', 'prod_brand']).prod_unit_price.mean()}).reset_index()
data = data[data["subcategory"] == "Lácteos, huevos y refrigerados"]
fig = px.treemap(data, path=['subcategory', 'prod_brand'], values='mean')
fig.update_layout(title_text='Map mean prices  20190609')
fig.show()
# Distribution by category and subcategory 
#data = df.groupby(['subcategory', 'prod_brand']).value_counts()[10:10].to_frame()
#sns.barplot(data.values, data.index, palette="husl")

data_brand=df['prod_brand'].value_counts()[:10].to_frame()
sns.barplot(data_brand['prod_brand'],data_brand.index,palette='husl')

dataproduct1 = df[df['prod_brand'] == 'ALPINA']
plt.title('Subcategoria Marcas ',size=3)
sns.boxplot(x="prod_unit_price", y="prod_brand", data=dataproduct1,palette='husl')


dataproduct2 = df[df['prod_brand'] == 'COLANTA']
plt.title('Subcategoria Marcas ',size=3)
sns.boxplot(x="prod_unit_price", y="prod_brand", data=dataproduct2,palette='husl')

# Different category distributions by standard deviation
data = df[df['date'] == 20200220]
sns.boxplot(x="prod_unit_price", y="prod_brand", data=data)
# Inspection of MARCAentre colanta y alpina 
df[(df['date'] == 20200220) & ((df['prod_brand'] == 'ALPINA') | (df['prod_brand'] == 'COLANTA') |  (df['prod_brand'] == 'COLANTA') )].head(5)
data = pd.DataFrame({'count' : df[(df['date'] == 20200220) & ((df['prod_brand'] == 'ALPINA'))].groupby(['prod_brand', 'prod_name']).size()}).reset_index()
fig = px.treemap(data, path=['prod_brand', 'prod_name'], values='count')
fig.show()
# Distribution by date and supermarket
datamarca = df[df['prod_brand'] == 'ALPINA']
data = datamarca.groupby(['date', 'prod_source']).size()
sns.barplot(data.values, data.index, palette='Blues')
plt.title('Distribución por fecha y supermecado de la subcategoria Marca AlPINA')
# Distribution by date and supermarket
datamarca1 = df[df['prod_brand'] == 'COLANTA']
data = datamarca.groupby(['date', 'prod_source']).size()
sns.barplot(data.values, data.index, palette='Blues')
plt.title('Distribución por fecha y supermecado de la subcategoria Marca COLANTA')

#Productos de la marca ALPINA vs precio 
datosVogue = df[(df['prod_brand'] == 'ALPINA')]
data1 = pd.DataFrame({'mean' : df[(df['prod_brand'] == 'ALPINA')].groupby(['prod_name']).prod_unit_price.mean()}).reset_index()
data1 = data1.sort_values(['mean'],ascending=False).reset_index(drop=True)
order = data1['prod_name'][:10]

data2 = df[(df['prod_brand'] == 'ALPINA')]
plt.figure(figsize = (5, 10))
plt.title('Productos con sus precios de la marca alpina',size=15)
sns.boxplot(x="prod_unit_price", y="prod_name", data=datosVogue, order= order ,palette='inferno')

print('Número de productos alpina:',datosVogue['prod_name'].nunique())

data1.head(10)

#Productos de la marca COLANTA vs precio 
datosVogue = df[(df['prod_brand'] == 'COLANTA')]
data1 = pd.DataFrame({'mean' : df[(df['prod_brand'] == 'COLANTA')].groupby(['prod_name']).prod_unit_price.mean()}).reset_index()
data1 = data1.sort_values(['mean'],ascending=False).reset_index(drop=True)
order = data1['prod_name'][:10]

data2 = df[(df['prod_brand'] == 'COLANTA')]
plt.figure(figsize = (5, 10))
plt.title('Productos con sus precios de la marca COLANTA',size=15)
sns.boxplot(x="prod_unit_price", y="prod_name", data=datosVogue, order= order ,palette='inferno')

print('Número de productos COLANTA:',datosVogue['prod_name'].nunique())

data1.head(10)