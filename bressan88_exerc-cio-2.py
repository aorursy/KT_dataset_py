# -------------------------

# IMPORTS

# -------------------------

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd
# -------------------------

# LEITURA DO ARQUIVO CSV

# -------------------------

df = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv')

df.head()
pd.DataFrame(df["Age"].value_counts())
f = df.sort_values(by=['Age'])

plt.figure(figsize=(10, 7))

sns.violinplot( x=f["Age"], y=f["Purchase"], linewidth=1)
plt.figure(figsize=(16, 5))

for indice, val in df["Product_ID"].value_counts().head(12).iteritems():

    plt.bar(indice, val, label = indice)

    plt.text(indice, val, val, va='bottom', ha='center') 

plt.title('Top 12 dos produtos mais comprados')

plt.show()
top = pd.DataFrame

for indice, val in df['Occupation'].value_counts().head(5).iteritems():    

    if top.empty:        

        top = df[df['Occupation'] == indice]

    else:

        top = top.append(df[df['Occupation'] == indice])

        

plt.figure(figsize=(20, 12))

plt.title('Distribuição de Gastos por Faixa Etária nas 5 ocupações mais frequentes')

sns.boxenplot(x = top['Occupation'],

              y = top['Purchase'], 

              hue = top['Age'],

              linewidth = 3)
purchase = df[df['Purchase'] > 9000]

sns.catplot(x='Marital_Status', y='Purchase', hue='Marital_Status',  margin_titles=True,

            kind='violin', col='Occupation', data=purchase, aspect=.4, col_wrap=7)
plt.figure(figsize=(5, 5))

df["Gender"].value_counts().plot(kind='bar', title='Homens x Mulheres')