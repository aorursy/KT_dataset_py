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
df = pd.read_csv("/kaggle/input/hospital-teste/hospital_teste.data")

df.head()

df.shape
df.head(21)
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
plt.figure(figsize=(15,10))

plt.hist(df['diagnostico'],bins=nanos, rwidth=0.8)

plt.title("Pessoas doentes x Pessoas saudáveis")

plt.xlabel("Doente")

plt.ylabel("Saudável")

plt.grid()

plt.show()
df.shape
print(df['manchas'].value_counts())

plt.pie(df['manchas'].value_counts() , labels=['pequenas','medias','grandes'])

plt.show()
print(df['sexo'].value_counts())

plt.pie(df['sexo'].value_counts(), labels=['feminino','masculino'])

plt.show()
print(df['diagnostico'].value_counts())

plt.pie(df['diagnostico'].value_counts(), labels=['doente','saudavel'])

plt.show()
df['temp'].mean()
df['temp'].min()
df['temp'].max()