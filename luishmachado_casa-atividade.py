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
casas = pd.read_csv('/kaggle/input/housedata/data.csv')

casas.head()                                                
casas.shape
print(casas['price'].max())

print(casas['price'].min())
print(casas['bathrooms'].max())
vista = casas['waterfront'].nunique()

print(vista)
cidades = casas['city'].nunique()

print(cidades)
cidadescasas = casas['city'].mode()

quantidade = casas['city'].value_counts()

print(cidadescasas,quantidade[0])
import matplotlib.pyplot as plt

numero = casas['city'].value_counts()

plt.hist(numero, bins=8, rwidth=0.9)
casas.corr()
andar = casas['floors'].value_counts()

plt.pie(andar)



#andar = pd.cut(x=casas['condition'], bins=[1,2,3,4,5]).value_counts()

#plt.pie(andar,labels=['andar1','andar','andar3','andar4','andar5'])
casas1 = casas[(casas['city']=='Seattle') & (casas['bedrooms']==3)]



print(casas1.std())



media = casas1.median()



print(media)
casas.corr()
casas['bathrooms'] =  casas['bathrooms'].astype(int)

contagem = casas['bathrooms'].value_counts().sort_index()

banheiros = contagem.index

print(contagem , banheiros)



plt.figure(figsize=(8,8))

for n, i in enumerate(contagem):

    plt.barh(n,i)

    plt.text(i+20,n,str(i))

plt.yticks([0,1,2,3,4,5,6,8])





#graf = casas['bathrooms'].value_counts()

#plt.hist(graf,bins = 8, rwidth=0.9)