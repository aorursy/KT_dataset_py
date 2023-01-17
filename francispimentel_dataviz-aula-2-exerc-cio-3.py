import pandas as pd
df = pd.read_csv('../input/BlackFriday.csv')
df.shape
df.dtypes
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
age_order = list(df['Age'].unique())

age_order.sort()



group_by_user = df.groupby(['User_ID', 'Age']).sum().reset_index()
sns.set()



fig = plt.figure(figsize=(14,16))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)



ax1.set_title('Valor gasto por produto x Idade', fontsize=20)

sns.violinplot(x='Age', y='Purchase', data=df, order=age_order, ax=ax1)

ax1.set_xlabel('Idade', fontsize=12)

ax1.set_ylabel('Valor do produto', fontsize=12)



ax2.set_title('Valor total gasto por cliente x Idade', fontsize=20)

sns.violinplot(x='Age', y='Purchase', data=group_by_user, order=age_order, ax=ax2)

ax2.set_xlabel('Idade', fontsize=12)

ax2.set_ylabel('Valor gasto', fontsize=12)

plt.show()
n = 10
df['Product_ID'].value_counts().head(n)
most_bought_items = list(df['Product_ID'].value_counts().head(n).index)

most_bought_items.reverse()
sns.set()



dimensions = (14, 8)

fig, ax = plt.subplots(figsize=dimensions)



sns.countplot(x='Product_ID', data=df, order=most_bought_items, ax=ax)

plt.title('Produtos mais comprados', fontsize=20)

plt.xlabel('Produto', fontsize=15)

plt.ylabel('Quantidade', fontsize=15)

plt.show()
# Considerando o total de gastos por cliente

group_by_user = df.groupby(['User_ID', 'Age', 'Occupation']).sum().reset_index()



only_most_frequent_occupations = group_by_user[group_by_user['Occupation'].isin(group_by_user['Occupation'].value_counts().head(5).index)]

only_most_frequent_occupations = only_most_frequent_occupations.sort_values(by='Age')



occupation_order = list(only_most_frequent_occupations['Occupation'].unique())

occupation_order.sort()
sns.set()

plt.figure(figsize=(16, 10))

sns.boxenplot(x='Occupation', y='Purchase', hue='Age', data=only_most_frequent_occupations, order=occupation_order)

plt.title('Gasto total por cliente - valores segregados por ocupação e idade\n', fontsize=18)

plt.xlabel('Ocupação do cliente', fontsize=13)

plt.ylabel('Valor gasto', fontsize=13)

plt.legend(loc=1, title='Idade', title_fontsize=13)

plt.ylim(0, 4000000)

plt.show()
df['Marital_Status'].unique()
# trazer todos os user_ID das compras maiores que 9000

purchases_over_9000 = df[df['Purchase'] > 9000]



# considerando apenas os clientes

customers_over_9000 = purchases_over_9000.drop_duplicates(subset='User_ID')
purchases_over_9000 = purchases_over_9000.groupby(['Occupation', 'Marital_Status']).count().reset_index()

customers_over_9000 = customers_over_9000.groupby(['Occupation', 'Marital_Status']).count().reset_index()
fig = plt.figure(figsize=(14,16))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)



# Values of each group

bars1 = np.array(purchases_over_9000[purchases_over_9000['Marital_Status'] == 0]['User_ID'])

bars2 = np.array(purchases_over_9000[purchases_over_9000['Marital_Status'] == 1]['User_ID'])

 

# Heights of bars1 + bars2

bars = np.add(bars1, bars2).tolist()

 

# The position of the bars on the x-axis

r = [int(i) * 3 for i in list(purchases_over_9000['Occupation'].unique())]

 

# Names of group and bar width

names = list(purchases_over_9000['Occupation'].unique())

barWidth = 2



ax1.bar(r, bars1, edgecolor='white', width=barWidth)

ax1.bar(r, bars2, bottom=bars1, edgecolor='white', width=barWidth)



ax1.set_xticks(r)

ax1.set_xticklabels(names)

ax1.set_xlabel('Ocupação do cliente', fontsize=15)

ax1.set_ylabel('Quantidade de produtos vendidos', fontsize=15)

ax1.set_title('Vendas acima de $9000 de acordo com perfil do cliente\n', fontsize=20)

ax1.legend(['0', '1'], fontsize=13, title='Estado civil', title_fontsize=15)



# Values of each group

bars1 = np.array(customers_over_9000[customers_over_9000['Marital_Status'] == 0]['User_ID'])

bars2 = np.array(customers_over_9000[customers_over_9000['Marital_Status'] == 1]['User_ID'])

 

# Heights of bars1 + bars2

bars = np.add(bars1, bars2).tolist()

 

# The position of the bars on the x-axis

r = [int(i) * 3 for i in list(customers_over_9000['Occupation'].unique())]

 

# Names of group and bar width

names = list(customers_over_9000['Occupation'].unique())

barWidth = 2



ax2.bar(r, bars1, edgecolor='white', width=barWidth)

ax2.bar(r, bars2, bottom=bars1, edgecolor='white', width=barWidth)



ax2.set_xticks(r)

ax2.set_xticklabels(names)

ax2.set_xlabel('Ocupação do cliente', fontsize=15)

ax2.set_ylabel('Quantidade de clientes', fontsize=15)

ax2.legend(['0', '1'], fontsize=13, title='Estado civil', title_fontsize=15)





plt.show()