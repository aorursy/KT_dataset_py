import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")

data.info()
data.corr() # ozellikler arasindaki baglanti degerlerini karsilastirmak icin tcorelasyon tablosunu verir
f,ax = plt.subplots(figsize = (7,7)) # resmin buyuklugunu ayarlar

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

#annot True uzerindeki degerleri yazar

#linewidth cizgilerin kalinligi

#fmt virgullu degerleri ayarlar



plt.show()
data.head() #datanin ilk 5.. degerlerini gosterir.

data.columns #datanin kolonlari gelir. 
data.home_score.plot(kind = 'line',color = 'g', label = 'home_score', linewidth =1, alpha = 0.5, grid = True, linestyle = '-.')

data.away_score.plot(color = 'r', label ='away_score', linewidth = 1, alpha = 0.5, grid = True,linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind="scatter", x="home_score", y="away_score", alpha= 1, color="blue")

plt.xlabel('home_score')

plt.ylabel('away_score')

plt.title('home_score and away_score Scatter Plot')

plt.show()
#histogram

data.home_score.plot(kind = 'hist',bins = 15,figsize = (6,6))#Speed kolonunu  #bins = cubuklarin kalinligini ayarlar

plt.show()
data.home_score.plot(kind = 'hist',bins = 10,figsize = (12,12))#Speed kolonunu  #bins = cubuklarin kalinligini ayarlar

plt.clf() # grafigi silmek icin clear icin
data = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")

series = data['home_score']        

print(type(series))     

data_frame = data[['home_score']]  

print(type(data_frame))

# pandas'da 3 adet veritipi var. 1) series 2)data_Frame

# series vektör şeklinde uzanır. tek boyutlu yapılardır.
x = data['away_score']>3    # away_score değeri 200'den büyük olan verileri x değişkenine atıyoruz.

data[x]
data[np.logical_and(data['away_score']>3, data['home_score']>3 )] # logical and fonksiyonu iki koşulunda sağlanması istenildiği koşullarda kullanılır.

# burada defense değeri 200'den büyük olsun aynı zamanda attack değeride 100'den büyük olsun diyoruz.

#numpy ile yaptik



# logical and fonksiyonuna alternatif olarak aşağıdaki yapı kullanılabilir.

data[(data['away_score']>3) & (data['home_score']>3)]