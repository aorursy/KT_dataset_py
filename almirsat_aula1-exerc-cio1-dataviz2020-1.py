from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('/kaggle/input/BR_eleitorado_2016_municipio.csv', delimiter=',')

df.dataframeName = 'eleitorado.csv'



df.head()
classification = [['UF' , 'Qualitativa Nominal'],

            ['Município' , 'Qualitativa Nominal'],['total_eleitores' , 'Quantitativa Discreta'],

            ['Feminino' , 'Quantitativa Discreta'],['Masculino' , 'Quantitativa Discreta']]

classification = pd.DataFrame(classification, columns=['Variavel' , 'Classificação'])

classification
uf = df['uf'].value_counts()

f_uf = df['uf'].value_counts(normalize=True)
freqr = pd.concat([uf,f_uf], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'], sort = False)

freqr
data = df.groupby(['uf'])['total_eleitores'].sum()

data = data.sort_values()[data > 0]

labels = data.keys().tolist()



plt.rcdefaults()

fig, ax = plt.subplots()

plt.xticks(rotation='vertical')



ax.bar(labels, data, align='center', ecolor='black', color='#ff4422')

ax.set_title('Quantidade de Eleitores por Região')



plt.show()