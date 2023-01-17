# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization



import matplotlib.pyplot as plt

import seaborn as sns



import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/steam-store-games/steam.csv")

df.head()
#Mengambil data berdasarkan tanggal rilis dan diambil tahunnya

df['release_year'] = pd.DatetimeIndex(df['release_date']).year

steamTahun = df.groupby(['release_year'])['appid'].count()



#Range tahun antara 2010 - 2029

steamTahunRange = steamTahun.loc[2010:2019]



#Membuat plot

plt.barh(steamTahunRange.index, steamTahunRange)

plt.xlabel("Jumlah Game")

plt.ylabel("Tahun")

plt.title("Jumlah Game Steam Tahun 2010 - 2019")



plt.show()
#Mengambil data berdasarkan bahasa

steamBahasa = df.groupby(['english'])['appid'].count()



#Membuat plot

plt.bar(steamBahasa.index, steamBahasa)

plt.xlabel("Bahasa")

plt.ylabel("Jumlah Game")

plt.title("Jumlah Game Steam dengan Bahasa Inggris")

bahasa = ['Lainnya', 'Inggris']

plt.xticks(steamBahasa.index, bahasa)



plt.show()
#Mengambil data berdasarkan harga

steamHarga = df.groupby(['price'])['appid'].count()



#Range harga

steamHargaRange = steamHarga.loc[10:20]



#Membuat plot

plt.barh(steamHargaRange.index, steamHargaRange)

plt.xlabel("Jumlah Game")

plt.ylabel("Harga")

plt.title("Jumlah Game Steam dengan Harga 10 - 20 Poundsterling")



plt.show()
#Ambil data berdasarkan positve_ratings

positifUrut = df.sort_values(by='positive_ratings', ascending=False).iloc[:10]

positifRating = positifUrut['positive_ratings']

positifNama = positifUrut['name']



#Membuat plot

plotFigure, plotRating = plt.subplots()

plotRating.barh(positifNama, positifRating)

plotRating.invert_yaxis()

plotFigure.suptitle('10 Game Terbaik')



plt.show()
plt.title('10 developer dengan jumlah game terbanyak')



sns.countplot(y='developer', data=df, order=df.developer.value_counts().iloc[:10].index)