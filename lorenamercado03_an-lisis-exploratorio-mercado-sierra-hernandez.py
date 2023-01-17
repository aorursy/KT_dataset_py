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
# Print the info of df
print(df.info())

# Print the shape of df
print(df.shape)
# Statistics for continuous variables
df.describe()
# Statistics for categorical variables
print(pd.DataFrame(df['date'].value_counts(dropna=False)))
df.describe(include=[np.object])
# Distribution by date and supermarket
data = df.groupby(['date', 'prod_source']).size()
sns.barplot(data.values, data.index, palette="Blues")
# For all the dates
plt.figure(figsize = (10, 10))
plt.subplots_adjust(hspace=0.1, wspace=1)
pal = sns.color_palette("husl", 20)

i = 1
for date in df['date'].unique():
    data = df[df['date'] == date].groupby(['subcategory']).size()      
    plt.subplot(2, 2, i)
    sns.barplot(data.values, data.index, palette=pal)
    i = i + 1

# Distribution by product in a specific date
#data = df[df['date'] == 20190609].groupby(['subcategory']).size()
#sns.barplot(data.values, data.index, palette="Blues")
# Different category distributions by size
data = pd.DataFrame({'count' : df[df['date'] == 20190709].groupby(['subcategory', 'tags']).size()}).reset_index()
fig = px.treemap(data, path=['subcategory', 'tags'], values='count')
fig.show()

#data = pd.DataFrame({'mean' : df[df['date'] == 20190709].groupby(['subcategory', 'tags']).prod_unit_price.mean()}).reset_index()
#fig = px.treemap(data, path=['subcategory', 'tags'], values='mean')
#fig.show()
#Mapa de calor de la categoria despensa
data = pd.DataFrame({'count' : df[(df['date'] == 20190709)& (df['subcategory']=='Despensa')].groupby(['tags']).size()}).reset_index()
fig = px.treemap(data, path=['tags'], values='count')
fig.show()
# Different category distributions by mean price
data = pd.DataFrame({'value' : df[df['date'] == 20190709].groupby(['subcategory']).prod_unit_price.mean()}).reset_index()
data = data.sort_values(['value'],ascending=False).reset_index(drop=True)
sns.barplot(data['value'], data['subcategory'], palette="Blues")
# Different tag distributions by mean price. More expensive
plt.figure(figsize = (5, 10))
data = pd.DataFrame({'value' : df[(df['date'] == 20190709) & (df['subcategory']== 'Despensa')].groupby(['tags']).prod_unit_price.mean()}).reset_index()
data = data.sort_values(['value'],ascending=False).reset_index(drop=True)
sns.barplot(data['value'], data['tags'], palette="Blues")
# Different category distributions by standard deviation
data = pd.DataFrame({'std' : df[df['date'] == 20190709].groupby(['subcategory', 'tags']).prod_unit_price.std()}).reset_index()
fig = px.treemap(data, path=['subcategory', 'tags'], values='std')
fig.show()
data = pd.DataFrame({'std' : df[(df['date'] == 20190709) & (df['subcategory']== 'Despensa')].groupby(['prod_name']).prod_unit_price.std()}).reset_index()
fig = px.treemap(data, path=['prod_name'], values='std')
fig.show()
# Different category distributions by standard deviation
data = df[df['date'] == 20190709]
sns.boxplot(x="prod_unit_price", y="subcategory", data=data)
# Different tags distributions without Vinos y Licores
data = pd.DataFrame({'mean' : df[(df['date'] == 20190709) & (df['subcategory'] == 'Despensa')].groupby(['tags']).prod_unit_price.mean()}).reset_index()
print(data)
data = data.sort_values(['mean'],ascending=False).reset_index(drop=True)
order = data['tags']

data = df[(df['date'] == 20190709) & (df['subcategory'] == 'Despensa')]
plt.figure(figsize = (10, 30))

sns.boxplot(x="prod_unit_price", y="tags", data=data, order= order)
df2=df[(df['tags']=='Salsas y vinagres') & (df['date']==20190709)]
df2=df2.sort_values(by=['prod_unit_price'], ascending=False)
df2[['prod_name','prod_unit_price']].head(20)
df2=df[(df['tags']=='Salsas y vinagres') & (df['date']==20190709)]
df2=df2.sort_values(by=['prod_unit_price'], ascending=False)
df2[['prod_name','prod_unit_price']].tail(20)
df2=df[(df['tags']=='Aceite') & (df['date']==20190709)]
df2=df2.sort_values(by=['prod_unit_price'], ascending=False)
df2[['prod_name','prod_unit_price']].tail(20)