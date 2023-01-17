# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
bitkidb = pd.read_csv('../input/Iris.csv') #Iris dbsini bitki değişkenine aktardık.

bitkidb.drop(['Id'],axis=1,inplace=True)
bitkidb.info() #Dataframe hakkında genel bilgileri aldık.
bitkidb.describe() #Dataframedeki bilgiler hakkındaki istatistikler.
f,ax= plt.subplots(figsize=(10,10))  # Tablomuzun boyutunu ayarladık.

sns.heatmap(bitkidb.corr(),annot=True,linewidth=.5,fmt='.2f',ax=ax)  #Oran-Orantı tablosu oluşturduk

plt.show()

# Tablodaki değerlerin birbirlerini hangi oranda etkilediğini bulduk.
setosa = bitkidb[bitkidb.Species == "Iris-setosa"]  #Setosa türündeki bitkiler için ayrı Db.

versicolor = bitkidb[bitkidb.Species == "Iris-versicolor"]  #Versicolor türündeki bitkiler için ayrı Db.

virginica = bitkidb[bitkidb.Species == "Iris-virginica"]  #Virginica türündeki bitkiler için ayrı Db.

#Databasedeki tüm bitkilerin tüm yaprak uzunluklarını tablolaştırdık.

bitkidb.plot(kind='line',grid=True,figsize=(9,9)) 

plt.xlabel('İndex Number')

plt.ylabel('Uzunluk (Cm)')

plt.show()
setosa.plot(kind='line',grid=True,figsize=(9,9))

plt.xlabel('İndex Number')

plt.ylabel('Uzunluk (Cm)')

plt.title('Setosa Türü')

plt.show()



versicolor.plot(kind='line',grid=True,figsize=(9,9))

plt.xlabel('İndex Number')

plt.ylabel('Uzunluk (Cm)')

plt.title('Versicolor Türü')

plt.show()



virginica.plot(kind='line',grid=True,figsize=(9,9))

plt.xlabel('İndex Number')

plt.ylabel('Uzunluk (Cm)')

plt.title('virginica Türü')

plt.show()