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
irisDF = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

irisDF.head()
irisDF.shape #bentuk/dimensi dataset (baris,kolom)

irisDF.columns
irisDF.isna().values.any()
irisDF[irisDF.duplicated(keep=False)] # menampilkan semua baris yang terduplikasi

#irisDF[irisDF.duplicated()] #menampilkan baris data sekunder yang terduplikasi
irisDF.duplicated().value_counts() #jumlah duplikasi
irisDF.drop_duplicates(inplace=True) #menghapus data sekunder yang duplikasi

irisDF.shape
irisDF.describe()
irisDF.corr()
import matplotlib.pyplot as plt

import seaborn as sns
sns.heatmap(data=irisDF.corr())
irisDF['species'].value_counts() #menghitung jumlah setiap species
irisDF['species'].value_counts().plot.bar()

plt.show()
sns.countplot(data=irisDF, x='species')

plt.tight_layout()
irisDF['species'].value_counts().plot.pie(autopct='%1.1f%%', labels=None, legend=True)# melihat jumlah data dari setiap species menggunakan persentase

plt.tight_layout()
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8)) # Tiap kolom akan di ploting pada tiap sub plotnya



irisDF['sepal_length'].plot.line(ax=ax[0][0])

ax[0][0].set_title('sepal_length')



irisDF['sepal_width'].plot.line(ax=ax[0][1])

ax[0][1].set_title('sepal_length')



irisDF['petal_length'].plot.line(ax=ax[1][0])

ax[1][0].set_title('sepal_length')



irisDF['petal_width'].plot.line(ax=ax[1][1])

ax[1][1].set_title('sepal_length')
irisDF.hist(figsize=(10,6), bins=10) 

plt.tight_layout()
irisDF.boxplot()

plt.tight_layout()
sns.scatterplot(x='sepal_length', y='sepal_width', data=irisDF, hue='species') # HUE= Mengelompokkan/mengkategori data berdasarkan species

#plt.scatter[sepal_length, sepal_width]

plt.tight_layout()
sns.violinplot(data=irisDF, y='species', x='sepal_length', inner ='quartile')

plt.tight_layout()