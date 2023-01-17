import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd
def read_file():

    return pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv')
df = read_file()

df.head()
f = df.sort_values(by=['Age'])

plt.figure(figsize=(10, 7))

sns.violinplot( x=f["Age"], y=f["Purchase"], linewidth=1)
plt.figure(figsize=(10, 8))

df["Product_ID"].value_counts().head(8).plot(kind='bar', title='Produtos mais comprados')
#Agrupando os dados por Usuário, idade e Ocupação

group_by_user = df.groupby(['User_ID', 'Age', 'Occupation']).sum().reset_index()



# Pegando somente as 5 ocupações mais frequentes

only_most_frequent_occupations = group_by_user[group_by_user['Occupation'].isin(group_by_user['Occupation'].value_counts().head(5).index)]



# Ordenando as 5 ocupações mais frequentes por idade

only_most_frequent_occupations = only_most_frequent_occupations.sort_values(by='Age')



# Criando uma lista com as ocupações mais frequentes

occupation_order = list(only_most_frequent_occupations['Occupation'].unique())

occupation_order.sort()
plt.figure(figsize=(18, 14))

sns.boxenplot(x=only_most_frequent_occupations['Occupation'], y=only_most_frequent_occupations['Purchase'], hue=only_most_frequent_occupations['Age'], linewidth=5)

plt.show()
purchase = df[df['Purchase'] > 9000]

sns.catplot(x='Marital_Status', y='Purchase', hue='Marital_Status',  margin_titles=True,

            kind='violin', col='Occupation', data=purchase, aspect=.4, col_wrap=7)