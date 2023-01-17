from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns
print(os.listdir('../input'))
df = pd.read_csv('../input/BlackFriday.csv')

df.head()
plt.figure(figsize=(15, 5))



fig = sns.violinplot(x=df['Age'].sort_values(), y=df['Purchase'].sort_values(), data=df)

fig.set_title('Valor gasto x Idade')

plt.ylabel('Valor gasto')

plt.xlabel('Idade')
# Filtrei pelos 15 primeiros mais vendidos (com quantidade > 8)

plt.figure(figsize=(15, 5))

mais_vendido = df.loc[(df.groupby('Product_ID')['Product_ID'].transform('count')) > 8]

mais_vendido = mais_vendido['Product_ID'].value_counts(ascending=False).head(15)

  

mais_vendido.plot.bar(width=0.3)



plt.ylabel('Quantidade')

plt.xlabel('Código Produto')

plt.title('Produto x Quantidade')
lst = []



top5 = df.groupby('Occupation').size().sort_values(ascending=False).head(5).index



for i in range(len(top5)):    

    lst.append(df[df['Occupation'] == top5[i]].groupby(by=['Age','Occupation']).sum())



df_o = pd.concat(lst)

df_o = df_o.sort_values(by=['Age','Occupation'],ascending=True)



df_o
x = df_o.groupby(['Age']).sum().index.get_level_values('Age').tolist()

y = df_o.groupby('Age').sum()['Purchase'].tolist()



plt.figure(figsize=(15, 5))



plt.xlabel('Idade')

plt.title('Idade com mais gasto')



plt.bar(x,y, color = 'darkblue')

plt.xticks(rotation='vertical')



plt.legend()



plt.show()
# montando os agrupamentos dos dados para exibição

df_r = df[df['Purchase'] > 9000].groupby(by=['Marital_Status','Occupation']).sum()

df_r
# exibindo agrupado por ocupação e estado civil

plt.figure(figsize=(20, 5))

plt.title('Compras entre Estado civil x Ocupação')



df_r.groupby(['Occupation','Marital_Status']).sum()['Purchase'].plot.bar(width=0.3)



plt.xlabel('Ocupação, Estado civil')
# exibindo agrupado apenas por ocupação

plt.figure(figsize=(20,5))

plt.title('Compras por Ocupação')



df_r.groupby(['Occupation']).sum()['Purchase'].plot.bar()

plt.xlabel('Ocupação')