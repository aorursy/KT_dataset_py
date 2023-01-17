# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import scipy.stats as stats

import matplotlib.pyplot as plt
import sklearn.datasets as df

dados = df.load_breast_cancer()
df.sample()
# antes de desenha o histograma:

idade_media = df['idade'].mean()                      

desvio_padrao = df['idade'].std()                     

str_std = "Desvio Padrão ="+str(round(desvio_padrao,2)) 

str_media = "idade Média ="+str(round(idade_media,2)) 



plt.hist(df['idade'],bins=8, rwidth=0.9)             

plt.title('Histograma da Idade dos Pacientes')

plt.xlabel('idade')

plt.ylabel('contagem')

plt.text(50, 150, str_std)                             

plt.text(50, 200, str_media)

plt.xlim(0, 100)

plt.ylim(0, 500)

plt.show()
nanos=df['diagnostico'].nunique()

diagnostico=df['diagnostico'].unique()

num_min=df['diagnostico'].min()

num_max=df['diagnostico'].max()

count_diagnostico=df['diagnostico'].value_counts().sort_index()



print(nanos)

print(diagnostico)

print(num_min)

print(num_max)

print(count_diagnostico)
diag= df['diagnostico'].value_counts()

print(diag)

plt.bar(0,diag[0], width=0.5)

plt.bar(1,diag[1], width=0.5)



plt.legend(['Doente','Saldavel']) 

plt.show()
df.shape
contagem = df["manchas"].value_counts()



taxa_p = contagem[0]/df["manchas"].count()*100

taxa_m = contagem[1]/df["manchas"].count()*100

taxa_g = contagem[1]/df["manchas"].count()*100



str_P = "Pequenas " + str(round(taxa_p,2))+"%"

str_M = "Médias " + str(round(taxa_m,2))+"%"

str_G = "Grandes " + str(round(taxa_g,2))+"%"



plt.figure(figsize=(5,5))

plt.pie(df["manchas"].value_counts(), labels=[str_P,str_M,str_G])

plt.show()
print(df['sexo'].value_counts())

plt.pie(df['sexo'].value_counts(), colors=['purple', 'blue'], labels=['feminino','masculino'])

plt.show()
print(df['diagnostico'].value_counts())

plt.pie(df['diagnostico'].value_counts(), labels=['doente','saudavel'])

plt.show()
plt.scatter(df.loc[101:150, 'temp'],

           df.loc[101:150, 'internacoes'],

           color='green')
df['temp'].mean()
df['temp'].min()
df['temp'].max()
import pandas as pd

breast_cancer = pd.read_csv("../input/breast-cancer/breast-cancer.csv")