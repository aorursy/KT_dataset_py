#imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv')

df.head()
#ordenação de ages

order_age = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
#tabela de frequencia de 'Age'

df['Age'].value_counts()
#Estatisticas de 'Purchase'

df['Purchase'].describe()
plt.figure(figsize = (10, 5))

plt.title('Age Versus Purchase')

ax = sns.violinplot(x='Age', y='Purchase', data=df, order = order_age)
#Produtos mais Comprados

top10 = df['Product_ID'].value_counts().head(10)

top10_prod = list(top10.index)

top10_prod
df_top10 = df[df['Product_ID'].isin(top10_prod)]
ax, fig = plt.subplots(figsize = (10, 5))

plt.title('Product Versus Purchase')

ax = sns.violinplot(x='Product_ID', y='Purchase', data=df_top10, order = top10_prod)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#Ocupações mais Frequentes

top5 = df['Occupation'].value_counts().head(5)

top5_occ = list(top5.index)

top5_occ
df_top5 = df[df['Occupation'].isin(top5_occ)]
plt.figure(figsize = (10, 18))



for i in range(len(top5_occ)):

  df_occ = df_top5[df['Occupation'] == top5_occ[i]]

  title = 'Purchase Amount by Age - Occupation nr. ' + str(top5_occ[i])

  plt.subplot(5, 1, (i + 1))

  plt.title(title)

  ax = sns.violinplot(x='Age', y='Purchase', data=df_occ, order = order_age)

plt.tight_layout()

plt.show()
df_9000 = df[df['Purchase'] > 9000]

df_9000.head()
ax, fig = plt.subplots(figsize = (15, 5))

plt.suptitle('Purchase Amount By Occupation and Marital Status')

plt.title('Purchases Over 9K')

ax = sns.violinplot(x='Occupation', y='Purchase', hue = 'Marital_Status', data= df_9000, split = True)
