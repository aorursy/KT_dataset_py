from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
df1= pd.read_csv('../input/cota_parlamentar_sp.csv', delimiter=';')

df1.dataframeName = 'cota_parlamentar_sp.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
parlamentar = df1['txnomeparlamentar'].value_counts().head(5)
partido = df1['sgpartido'].value_counts()
df_bubble = pd.DataFrame([], columns=list(['partido','count', 'total']))

for i,v in partido.iteritems():    

    soma = float(0)

    row = df1[df1['sgpartido'] == i]

    for r in row.itertuples(index=True, name='Pandas'):

        soma = soma  + getattr(r, 'vlrdocumento')

    df_bubble = df_bubble.append({'partido': i, 'count':float(v), 'total': soma}, ignore_index=True)
import matplotlib.pyplot as plt

import numpy as np

 

plt.figure(figsize=(16, 6))

# create data

x = df_bubble['total']

y = df_bubble['partido']

z = df_bubble['count']

 

# Change color with c and alpha

plt.scatter(x, y, s=z/10,  alpha=0.5)

df1['txtfornecedor'].value_counts().head(5)
import seaborn as sns

plt.figure(figsize=(16, 6))

daf = df1[(df1['vlrdocumento'] > 0)]

# plot of 2 variables

sns.violinplot(x="sgpartido", y="vlrdocumento", data=daf,

            palette="Set2")

#sns.swarmplot(x="", y="method", data=planets,

#              size=2, color=".3", linewidth=0)



#Renan

cota = df1

cota_total = pd.DataFrame(cota.groupby(['sgpartido'])['vlrdocumento'].sum().sort_values(ascending=False))

cota_total2 = cota_total.head(5)

#cota.groupby('nulegislatura')['vlrdocumento'].sum().sort_values(ascending=False)

cota_por_ano = pd.DataFrame(cota.groupby('numano')['vlrdocumento'].sum())

#partido_valor.groupby(['sgpartido']).sum()

cota_por_ano

cota_total
cota_total.plot(kind='bar', title='Gastos Partidos - Completo')
cota_total2
cota_total2.plot(kind='pie', title='Maiores Gastos Partidos - Top5', subplots=True)
cota_por_ano
plt.title("Evolução Gastos dos Partidos", loc='center', fontsize=12, fontweight=0, color='black')

plt.xlabel("Ano")

plt.ylabel("Gasto")

plt.plot(cota_por_ano)
df_dep_partido = df1.groupby(['sgpartido'])['nudeputadoid'].nunique().reset_index(name='nunique').sort_values(by='nunique', ascending=False).reset_index(drop=True)

df_dep_partido['vlrdocumento'] = cota_total.sort_index()['vlrdocumento'].tolist()
sns.set(style="whitegrid")

ax = sns.barplot(x="sgpartido", y="nunique", data=df_dep_partido, )



ax.set_xlabel('Partidos')

ax.set_ylabel('Quantidade de Deputados')



for item in ax.get_xticklabels():

    item.set_rotation(90)

    

for i in range(len(df_dep_partido['sgpartido'])):

    plt.text(x = i - 0.3 , y = df_dep_partido.loc[i,'nunique'] + 1 , s = df_dep_partido.loc[i,'nunique'], size = 8, color='Blue')



plt.show()
df_partido_ano = df1.groupby(['sgpartido', 'numano'])['vlrdocumento'].sum().reset_index(name='sum').sort_values(by='sgpartido').reset_index(drop=True)

df_partido_ano.head()
valor_anual_dep  = []

for index, row in df_partido_ano.iterrows():

    valor_anual_dep.append(round(int(row['sum']) / int(df_dep_partido[df_dep_partido['sgpartido'] == row['sgpartido']]['nunique'])))



df_partido_ano['total_por_deputado'] = valor_anual_dep

df_partido_ano.head()