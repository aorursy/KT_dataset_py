import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
bf = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

bf.head()
bf['Age'].value_counts()
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))

plt.title('Qtd de Compras x Idades')

sns.violinplot(y = bf["Purchase"],

               x = bf["Age"].sort_values(ascending=True),

               scale="count")
plt.figure(figsize=(16, 6))

for indice, val in bf["Product_ID"].value_counts().head(10).iteritems():

    plt.bar(indice, val, label = indice)

    plt.text(indice, val, val, va='bottom', ha='center') 

plt.title('Top 10 - Produtos mais Comprados')

plt.show()
top = pd.DataFrame

for indice, val in bf['Occupation'].value_counts().head(5).iteritems():    

    if top.empty:        

        top = bf[bf['Occupation'] == indice]

    else:

        top = top.append(bf[bf['Occupation'] == indice])

        

plt.figure(figsize=(20, 10))

plt.title('Distribuição de Gastos por Faixa Etária')

sns.boxenplot(x = top['Occupation'],

              y = top['Purchase'], 

              hue = top['Age'],

              linewidth = 5)
sns.catplot(x = 'Marital_Status',

            y = 'Purchase',

            hue = 'Marital_Status',

            margin_titles = True,

            kind = 'point',

            col = 'Occupation',

            data = bf[bf['Purchase'] > 9000],

            aspect = .4,

            col_wrap = 7)