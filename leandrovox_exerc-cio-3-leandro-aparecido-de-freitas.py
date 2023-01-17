from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df1 = pd.read_csv('../input/BlackFriday.csv', delimiter=',')

df1.head(10)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

sns.set(style="whitegrid")

ax = sns.violinplot(x=df1['Age'], y=df1['Purchase'], palette='Set2')
produtos = df1['Product_ID'].value_counts().head(10)

plt.figure(figsize=(16, 6))

for i, v in produtos.iteritems():

    plt.bar(i, v, label = i)

    plt.text(i, v, v, va='bottom', ha='center')    

    

plt.title('Produtos mais comprados')

plt.show()
occupation = df1['Occupation'].value_counts().head(5)
aux = pd.DataFrame

for i, v in occupation.iteritems():    

    if aux.empty :        

        aux = df1[df1['Occupation'] == i]

    else:

        aux = aux.append(df1[df1['Occupation'] == i])

plt.figure(figsize=(20, 10))

sns.boxenplot(x=aux['Occupation'], y=aux['Purchase'], hue=aux['Age'])
purchase = df1[df1['Purchase'] > 9000]
plt.figure(figsize=(16, 6))

sns.catplot(x='Marital_Status', y='Purchase', hue='Marital_Status', margin_titles=True,

            kind="box", col='Occupation', data=purchase, aspect=.4, col_wrap=7,)