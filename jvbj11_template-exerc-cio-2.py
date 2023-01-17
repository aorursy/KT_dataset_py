import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
bf = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

bf.head(10)
plt.figure(figsize=(15, 4))

plt.title('Total de Compras por grupo de idade')

sns.violinplot(y=bf["Purchase"],x=bf["Age"].sort_values(ascending=True),scale="count")
plt.figure(figsize=(8, 5))

bf["Product_ID"].value_counts().head(8).plot(kind='bar', title='Os 8 produtos mais comprados')
# Agrupando informações

gp_u = bf.groupby(['User_ID', 'Age', 'Occupation']).sum().reset_index()



# 5 mais frequentes

freq5 = gp_u[gp_u['Occupation'].isin(gp_u['Occupation'].value_counts().head(5).index)]



# 5 frequentes com base na idade

freq5_idade = freq5.sort_values(by='Age')



#Tamanho do gráfico

plt.figure(figsize=(20, 12))



#Montagem do gráfico

sns.boxplot(x=freq5_idade['Occupation'], y=freq5_idade['Purchase'], hue=freq5_idade['Age'], linewidth=5)
sns.catplot(x='Marital_Status',y='Purchase',hue = 'Marital_Status',kind = 'point',col = 'Occupation',aspect = .4,col_wrap = 7, data = bf[bf['Purchase'] > 9000])