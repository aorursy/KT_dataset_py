

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import warnings #to ignore any warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv',encoding = "ISO-8859-1")

df.head()
df.tail()
df.info()
df.columns  = df.columns.str.lower()

df.columns = df.columns.str.replace(' ','_')

df.head()
df.drop(['area_code','item_code','element_code','year_code','flag'], axis = 1, inplace = True)

df.head()
df.duplicated().any()
df1 = df.area.value_counts().head(10).reset_index()

df1 = df1.rename(columns = {'index':'Countries','area':'No. of Fertilizers'})

fig = px.bar(df1, x = 'No. of Fertilizers', y= 'Countries', orientation='h', title = 'Top 10 Countries with Most Fertilizers used')

fig.update_layout(autosize = False)

fig.show()
India = df[df.area == 'India']

India.head()
India.shape
India.element.value_counts()
plt.figure(figsize=(20,7.5))

sns.countplot(x = 'item', data=India, hue = 'element')

plt.title('Different Uses of Fertilizers', size = 18)



plt.xticks(rotation = 90)

plt.show()

df_prod = India[India.element == 'Production']



#drop irrelevant columns from this dataframe : area (since we know we are analysing Indian market) and element (since we are analysing only one element)



df_prod.drop(['area','element'],axis = 1, inplace = True)



df_prod.head()


fig = px.area(df_prod, x = 'year', y = 'value', color = 'item',title = 'Fertilizers produced over the years 2002-2017 in India', line_group = 'item')

fig.update_layout(legend_orientation = 'h', autosize = False)

fig.show()
df_agr = India[India.element == 'Agricultural Use']



#drop irrelevant columns from this dataframe : area (since we know we are analysing Indian market) and element (since we are analysing only one element)



df_agr.drop(['area','element'],axis = 1, inplace = True)



df_agr.head()
fig = px.area(df_agr, x = 'year', y = 'value', color = 'item',title = 'Fertilizers used over the years 2002-2017 in India', line_group = 'item')

fig.update_layout(legend_orientation = 'h',autosize = False, height = 600, width = 800)

fig.show()


#getting all the data where element is export quantity

df_equan = India[India.element == 'Export Quantity']

df_equan.head()


top_5_exp = df_equan.groupby('item')['value'].sum().sort_values(ascending = False).head().reset_index()

fig = px.bar(top_5_exp, x = 'value', y = 'item', orientation = 'h', title = 'Top 5 Fertilizers Exported')

fig.show()



#getting the fertilizers name

top_5_exp = top_5_exp['item']



#list to store the dataframes created for top 5 fertilizers

df_to_concat = []



for i in top_5_exp:

    items = df_equan[df_equan.item == i]

    df_to_concat.append(items)

    

result = pd.concat(df_to_concat)



#plotting line graph

fig_equan = px.line(result, x = 'year', y = 'value', color = 'item', title = 'Top 5 Fertilizers exported from 2002-2017')

fig_equan.update_layout(legend_orientation = 'h', height= 600)

fig_equan.show()

#getting all the data where element is import quantity

df_iquan = India[India.element == 'Import Quantity']

df_iquan.head()
#getting top 5 fertilizers that were imported

top_5_imp = df_iquan.groupby('item')['value'].sum().sort_values(ascending = False).head().reset_index()

fig = px.bar(top_5_imp, x = 'value', y = 'item', orientation = 'h', title = 'Top 5 Fertilizers Imported')

fig.show()


#getting the fertilizers name

top_5_imp = top_5_imp['item']



#list to store the dataframes created for top 5 fertilizers

df_to_concat = []



for i in top_5_imp:

    items = df_equan[df_equan.item == i]

    df_to_concat.append(items)

    

result = pd.concat(df_to_concat)



#plotting line graph

fig_equan = px.line(result, x = 'year', y = 'value', color = 'item', title = 'Top 5 Fertilizers imported from 2002-2017')

fig_equan.update_layout(legend_orientation = 'h', height = 600)

fig_equan.show()

df_urea = India[(India.item == 'Urea') & (India.unit == 'tonnes')]

df_urea.head()
plt.figure(figsize=(10,5))

df_urea.groupby('element')['value'].sum().plot(kind = 'bar')

plt.title('Urea - Use vs Import vs Export vs Production', size = 18)

plt.xlabel('Different Uses', size = 15)

plt.ylabel('Amount of Urea used', size = 15)

plt.show()