import pandas as pd

import numpy as ny

import matplotlib.pyplot as mlt

import seaborn as sbn
ler = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

ler.head()
sbn.set(style="whitegrid")

mlt.figure(figsize=(12, 6))

mlt.title('Compras x Idades')

sbn.violinplot(y = ler["Purchase"],

               x = ler["Age"].sort_values(ascending=True),

               scale="count")
mlt.figure(figsize=(16, 6))

for indice, val in ler["Product_ID"].value_counts().head(5).iteritems():

    mlt.bar(indice, val, label = indice)

    mlt.text(indice, val, val, va='bottom', ha='center') 

mlt.title('Produtos mais Comprados')

mlt.show()
top = pd.DataFrame

for indice, val in ler['Occupation'].value_counts().head(5).iteritems():    

    if top.empty:        

        top = ler[ler['Occupation'] == indice]

    else:

        top = top.append(ler[ler['Occupation'] == indice])

        

mlt.figure(figsize=(20, 10))

mlt.title('Distribuição de Gastos por Faixa Etária')

sbn.boxenplot(x = top['Occupation'],

              y = top['Purchase'], 

              hue = top['Age'],

              linewidth = 5)
sbn.catplot(x = 'Marital_Status',

            y = 'Purchase',

            hue = 'Marital_Status',

            margin_titles = True,

            kind = 'point',

            col = 'Occupation',

            data = ler[ler['Purchase'] > 7000],

            aspect = .4,

            col_wrap = 7)