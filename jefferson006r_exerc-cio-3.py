%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



bf = pd.read_csv('../input/BlackFriday.csv', delimiter=',')

bf.drop(['Product_Category_1', 'Product_Category_2', 'Product_Category_3'], axis = 1, inplace = True)

pd.isnull(bf).sum()
plt.figure(figsize=(10, 6))

sns.set_style('whitegrid')

sns.violinplot(x='Age', y='Purchase', cut=0, scale="count", data=bf.sort_values(by=['Age']))
pd.DataFrame(bf["Product_ID"].value_counts())

plt.figure(figsize=(10, 6))

bf["Product_ID"].value_counts().head(10).plot(kind='bar', title='Produtos mais comprados')
df = bf['Occupation'].value_counts().head(5)

w = pd.DataFrame

for i, v in df.iteritems():    

    if w.empty :        

        w = bf[bf['Occupation'] == i]

    else:

        w = w.append(bf[bf['Occupation'] == i])

        

plt.figure(figsize=(20, 10))

sns.boxenplot(x=w['Occupation'], y=w['Purchase'], hue=w['Age'], linewidth=5)
valor = bf[bf['Purchase'] > 9000]

sns.catplot(x='Marital_Status', y='Purchase', hue='Marital_Status',  margin_titles=True,

            kind='violin', col='Occupation', data=valor, aspect=.4, col_wrap=7,)