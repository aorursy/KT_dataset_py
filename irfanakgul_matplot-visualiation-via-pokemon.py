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
data = pd.read_csv("../input/pokemon/Pokemon.csv")

data

data.describe()

att = data.Attack.mean()

dfs = data.Defense.mean()

best1 = data.Defense & data.Attack > 150

best11 = data[best1]

best111 = best11.sort_values(by='Defense', ascending=False).head(1)

print(best111)



data.corr() # ozellikler arasindaki baglanti degerlerini karsilastirmak icin tcorelasyon tablosunu verir
f,ax = plt.subplots(figsize = (18,18)) # resmin buyuklugunu ayarlar

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

#annot True uzerindeki degerleri yazar

#linewidth cizgilerin kalinligi

#fmt virgullu degerleri ayarlar



plt.show()
data.head() #datanin ilk 5.. degerlerini gosterir.

data.columns #datanin kolonlari gelir. 

data.Speed.plot(kind = 'line',color = 'g', label = 'Speed', linewidth =1, alpha = 0.5, grid = True, linestyle = '-.')

data.Defense.plot(color = 'r', label ='Defense', linewidth = 1, alpha = 0.5, grid = True,linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind="scatter", x="Attack", y="Defense", alpha= 0.5, color="blue")

plt.xlabel('Attack')

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')

plt.show()
#histogram

data.Speed.plot(kind = 'hist',bins = 10,figsize = (12,12))#Speed kolonunu  #bins = cubuklarin kalinligini ayarlar

plt.show()
data.Speed.plot(kind = 'hist',bins = 10,figsize = (12,12))#Speed kolonunu  #bins = cubuklarin kalinligini ayarlar

plt.clf() # grafigi silmek icin clear icin
#Dictionary listelerden daha hızlı.

dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys()) #key değerlerini ekrana yazdırıyor.

print(dictionary.values()) #values değerlerini ekrana yazdırıyor.

dictionary['spain'] = "barcelona"    # spain anahtarının değerini güncelledik.

print(dictionary)

dictionary['france'] = "paris"       # yeni bir key ve value ekledik.

print(dictionary)

del dictionary['spain']              # spain anahtarını sildik.

print(dictionary)

print('france' in dictionary)       

dictionary.clear()                   

print(dictionary)
data = pd.read_csv("../input/pokemon/Pokemon.csv")

series = data['Defense']        

print(type(series))     

data_frame = data[['Defense']]  

print(type(data_frame))

# pandas'da 3 adet veritipi var. 1) series 2)data_Frame

# series vektör şeklinde uzanır. tek boyutlu yapılardır.

#filtrelemeler

x = data['Defense']>200    # Defans değeri 200'den büyük olan verileri x değişkenine atıyoruz.

data[x]
data[np.logical_and(data['Defense']>200, data['Attack']>100 )] # logical and fonksiyonu iki koşulunda sağlanması istenildiği koşullarda kullanılır.

# burada defense değeri 200'den büyük olsun aynı zamanda attack değeride 100'den büyük olsun diyoruz.

#numpy ile yaptik



# logical and fonksiyonuna alternatif olarak aşağıdaki yapı kullanılabilir.

data[(data['Defense']>200) & (data['Attack']>100)]
threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10, ["speed_level", "Speed"]]
data.describe()
print(data["Type 1"].value_counts(dropna = False))

#type 1 sutunundaki degerlerin kactane oldugunu yazar

#dropna false ile bos degerleri almadik

#boxplot cizimi

data.boxplot(column="Attack", by="Legendary")
data_new = data.head()

data_new
melted = pd.melt(frame = data_new, id_vars = "Name", value_vars= ["Attack","Defense"])

melted

#sadece "Attack" ve "Defense" sutunlarini alir ve alt alta ekler. 

#coklu sutunlarda belirli sutunlarda daha kolay calismak icin kullanilir.

melted.pivot(index="Name", columns="variable", values="value")

#sadece attack ve defanse sutunlarindaki ilk bes elemani alir(head yaptik o yuzden ilk bes)
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row

#iki DF yi birlestirdik

data1 = data['Attack'].head()

data2= data['Defense'].head()

data3= data['Name'].head()

conc_data_col = pd.concat([data1,data2,data3],axis =1) # axis = 0 : adds dataframes in row

conc_data_col

#3 df yi birlestirdi
# lets convert object(str) to categorical and int to float.

#datanin type ni degistirir. donusum yapar

data['Type 1'] = data['Type 1'].astype('category')

#data type 1 in tipini category tipine cevirdi

data['Speed'] = data['Speed'].astype('float')

#data "speed" in tipini "float" tipine cevirdi

data.dtypes
