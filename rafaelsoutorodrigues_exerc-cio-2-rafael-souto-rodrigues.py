%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
b_friday = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

b_friday.head()
pd.DataFrame(b_friday["Age"].value_counts())
sns.set(style="whitegrid")

plt.figure(figsize=(16, 6))

plt.title('Compras x Grupo de Idades')

sns.violinplot(y = b_friday["Purchase"],

               x = b_friday["Age"].sort_values(ascending=True),

               scale="count")
plt.figure(figsize=(16, 6))

for index, valor in b_friday["Product_ID"].value_counts().head(10).iteritems():

    plt.bar(index, valor, label = index)

    plt.text(index, valor, valor, va='bottom', ha='center') 

plt.title('Top 10 - Produtos mais Comprados')

plt.show()
top_5 = pd.DataFrame

for index, valor in b_friday['Occupation'].value_counts().head(5).iteritems():    

    if top_5.empty:        

        top_5 = b_friday[b_friday['Occupation'] == index]

    else:

        top_5 = top_5.append(b_friday[b_friday['Occupation'] == index])

        

plt.figure(figsize=(20, 10))

plt.title('Distribuição de Gastos por Faixa Etária nas 5 Ocupações mais Frequentes')

sns.boxenplot(x = top_5['Occupation'],

              y = top_5['Purchase'], 

              hue = top_5['Age'],

              linewidth = 5)
sns.catplot(x = 'Marital_Status',

            y = 'Purchase',

            hue = 'Marital_Status',

            margin_titles = True,

            kind = 'point',

            col = 'Occupation',

            data = b_friday[b_friday['Purchase'] > 9000],

            aspect = .4,

            col_wrap = 7)