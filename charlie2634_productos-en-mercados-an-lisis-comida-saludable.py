!pip install plotly==4.5.2
# Helpful packages to load in 

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
# Leyendo excel.
df = pd.read_excel('/kaggle/input/productos-consumo-masivo/output - Kaggle.xlsx', decimal=',')

# Realizando filtrado por categoria pasabocas.
df = df[(df['subcategory'] == 'Pasabocas') & (df['tags'] != 'Mermelada y Cremas de untar')]

# Filtrando columnas para tomar solo las relevantes para el analisis actual.
df = df[['prod_name','prod_brand', 'tags', 'prod_unit_price', 'prod_source']]

# Imprimiendo primeros 3 registros.
df.head(3)
# Print the info of df
print("-------")
print(df.info())

# Print the shape of df
print("-------")
print(df.shape)
# Estadisticas para proveedor 'VERDE'
print("ESTADISTICAS PARA PROVEEDOR 'VERDE'")
print(df[(df['prod_source'] == 'VERDE')].describe())
# Estadisticas para proveedor 'AMARILLO'
print("ESTADISTICAS PARA PROVEEDOR 'AMARILLO'")
print(df[(df['prod_source'] == 'AMARILLO')].describe())
# Cantidad de productos ofrecidos por proveedor y categoria.
data = pd.DataFrame({'count' : df.groupby(['tags','prod_source']).size()}).reset_index()
fig = px.treemap(data, path=[ 'prod_source', 'tags'], values='count')
fig.show()
# BoxPlot de precios para categoria 'VERDE'.
sns.boxplot(x="prod_unit_price", y="tags", data=df[(df['prod_source'] == 'VERDE')])
# BoxPlot de precios para categoria 'AMARILLO'.
sns.boxplot(x="prod_unit_price", y="tags", data=df[(df['prod_source'] == 'AMARILLO')])
# Cantidad de productos ofrecidos dependiendo de su categoria (Saludable no saludable) y su marca.
data = df
data["type"] = np.where(((df['tags'] == 'Frutos Secos') | (df['tags'] == 'Galletas dietéticas y saludables')), 'Saludable', 'NO Saludable')
data = pd.DataFrame({'count' : df.groupby(['type','tags','prod_brand']).size()}).reset_index()
fig = px.treemap(data, path=[ 'type', 'tags','prod_brand'], values='count')
fig.show()
# Porcentaje de productos saludables y no saludables ofrecidos por cada marca.
data = df
data["type"] = np.where(((df['tags'] == 'Frutos Secos') | (df['tags'] == 'Galletas dietéticas y saludables')), 'Saludable', 'NO Saludable')
data = pd.DataFrame({'count' : df.groupby(['type','tags','prod_brand']).size()}).reset_index()
fig = px.treemap(data, path=[ 'prod_brand', 'type'], values='count')
fig.show()
import numpy as np
import matplotlib.pyplot as plt
data.dtypes
tipo = df[['type', 'prod_brand']].groupby('type').count().round(2)
tipo
fig, axes = plt.subplots(figsize=(6,4), dpi=80)
plt.bar(tipo.index, height=tipo.prod_brand)
plt.title('Productos por type');
df[(df['type'] == 'NO Saludable')].groupby("prod_brand").prod_brand.count().sort_values(ascending=False)[:7].plot.barh(color='red',alpha=0.5)
plt.grid(color='#95a5a6', linestyle= '--', linewidth=2, axis='x', alpha=0.2)
plt.show()
df[(df['type'] == 'Saludable')].groupby("prod_brand").prod_brand.count().sort_values(ascending=False)[:7].plot.barh(color='green',alpha=0.5)
plt.grid(color='#95a5a6', linestyle= '--', linewidth=2, axis='x', alpha=0.2)
plt.show()
df3=df[df.prod_brand.isin(['INSUALIMENTOS', 'GULLON','LA ESPECIAL','MANITOBA','FRITO LAY','JBO MP','HATSU'])]
df3[(df3['type'] == 'NO Saludable')].groupby("prod_brand").prod_brand.count().sort_values(ascending=False)[:7].plot.barh(color='yellow',alpha=0.5)
plt.grid(color='#95a5a6', linestyle= '--', linewidth=2, axis='x', alpha=0.2)
plt.show()