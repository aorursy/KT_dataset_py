from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

cota = pd.read_csv('../input/cota_parlamentar_sp.csv', delimiter=';')

cota.dataframeName = 'cota_parlamentar_sp.csv'

nRow, nCol = cota.shape

print(f'There are {nRow} rows and {nCol} columns')
cota.head(10)
#partido_valor = pd.DataFrame()

cota_negativa = pd.DataFrame()

cota_positiva = pd.DataFrame()

cota_negativa = cota[cota['vlrdocumento'] < 0]

cota_positiva = cota[cota['vlrdocumento'] > 0]

#cota_negativa['reembolso'] = pd.DataFrame()

cota_negativa['tipo'] = 'Reembolso'

cota_positiva['tipo'] = 'Gasto'

#partido_valor['sgpartido'] = cota['sgpartido']

#partido_valor['vlrdocumento'] = cota['vlrdocumento']

#partido_valor['nulegislatura'] = cota['nulegislatura']

#cota['nulegislatura'].value_counts()

#cota.groupby(['nulegislatura']).sum()

cota_total = pd.DataFrame(cota.groupby(['sgpartido'])['vlrdocumento'].sum().sort_values(ascending=False))

cota_total2 = cota_total.head(5)

cota_total3 = cota_total.tail(5)

#cota.groupby('nulegislatura')['vlrdocumento'].sum().sort_values(ascending=False)

cota_por_ano = pd.DataFrame(cota.groupby('numano')['vlrdocumento'].sum())

cota_negativa_por_ano = pd.DataFrame(cota_negativa.groupby('numano')['vlrdocumento'].sum())

#partido_valor.groupby(['sgpartido']).sum()

cota_por_ano

cota_positiva

new_cota = pd.DataFrame()

new_cota = pd.concat([cota_positiva, cota_negativa])

pd.DataFrame(new_cota['tipo'].value_counts())
cota_total
cota_total.plot(kind='bar', title='Gastos Partidos - Completo')
#cota_total2.plot(kind='pie', title='Maiores Gastos Partidos - Top5', subplots=True)

# Data to plot

labels = cota_total2.index

sizes = cota_total2['vlrdocumento']

#colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0.1, 0, 0, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.show()
#cota_total2.plot(kind='pie', title='Maiores Gastos Partidos - Top5', subplots=True)

# Data to plot

labels = cota_total3.index

sizes = cota_total3['vlrdocumento']

#colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0.1, 0, 0, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.show()
plt.title("Evolução Gastos dos Partidos", loc='center', fontsize=12, fontweight=0, color='black')

plt.xlabel("Ano")

plt.ylabel("Gasto")

plt.plot(cota_por_ano)
plt.title("Evolução Gastos dos Partidos", loc='center', fontsize=12, fontweight=0, color='black')

plt.xlabel("Ano")

plt.ylabel("Gasto")

plt.plot(cota_negativa_por_ano)
cota_neg_ano_mes = pd.DataFrame(cota_negativa.groupby(['numano','nummes'])['vlrdocumento'].sum())
cota.describe()
pd.DataFrame(cota.groupby(['numano','nummes'])['vlrdocumento'].sum())
cota_negativa.describe()
val1 = pd.DataFrame(new_cota['tipo'].value_counts())

val1['perc'] = val1['tipo'] / np.cumsum(val1['tipo'],axis=0)

val1


#new_cota['reembolso'].value_counts().plot(kind='pie', title='Valores Positivos e Negativos', subplots=True)

# Data to plot

#labels = val1.index

#sizes = val1['tipo']

#colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

#explode = (0.1, 0)  # explode 1st slice



# Plot

#plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)

#autotexts.set_color('white')

#plt.axis('equal')

#plt.show()



plt.figure(figsize=(6, 6))

for i, v in val1['tipo'].iteritems():

    plt.bar(i, v, label = i)

    plt.text(i, v, v, va='bottom', ha='center')    

    

plt.title('Gastos x Reembolso')

plt.show()
# style

plt.style.use('seaborn-darkgrid')

 

# create a color palette

palette = plt.get_cmap('Set1')

 

# multiple line plot

num=0

for column in cota_por_ano.drop('numano', axis=1):

    num+=1

    plt.plot(cota_por_ano['numano'], cota_por_ano[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

 

    # Add legend

    plt.legend(loc=2, ncol=2)

 

# Add titles

plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')

plt.xlabel("Time")

plt.ylabel("Score")
cota_por_ano