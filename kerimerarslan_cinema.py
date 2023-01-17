# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/tmdb_5000_movies.csv')

data.info()

data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.head(15)
data.columns
data.revenue.plot(kind = 'line', color = 'g',label = 'revenue',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.popularity.plot(color = 'r',label = 'popularity',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x')              

plt.ylabel('y')

plt.title('Line Plot')            

plt.show()

data.plot(kind='scatter', x='revenue', y='popularity',alpha = 0.5,color = 'red')

plt.xlabel('revenue')              

plt.ylabel('popularity')

plt.title('revenue popularity Scatter Plot') 
data.revenue.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.vote_count.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.runtime.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
#çizdirdiğini temizlemek için

data.runtime.plot(kind = 'hist',bins = 50)

plt.clf()
series = data['original_language']        #series

print(type(series))

data_frame = data[['original_language']]  #data frame

print(type(data_frame))
#pandas ile data frame filtreleme

x = data['revenue']>200     

data[x]
#logical_and ile iki verinin kesişimlerini bulma

data[np.logical_and(data['revenue']>200, data['runtime']>200 )]
#aynı işi & ile yapar

data[(data['revenue']>200) & (data['popularity']>100)]
#sinema filmlerinin gelirlerini ortalamanın üstünde mi altında mı olduğunu görmek için 

#revenue verisini kullanarak yeni bir sütun oluşturup değerleri girdirdim

threshold = sum(data.revenue)/len(data.revenue)

data["revenue_level"] = ["high" if i > threshold else "low" for i in data.revenue]

data.loc[:2000,["title","revenue_level","revenue"]] 
data = pd.read_csv('../input/tmdb_5000_movies.csv')

data.head()  # ilk 5 satır
#datadaki son 5 satır

data.tail()
#datadaki satır sütun sayılarını verir

data.shape
#datadali sütunlar

data.columns
#data hakkında bilgi

data.info()
print(data['original_language'].value_counts(dropna =False))
#verileri max min vb. değerlerini gösterir

data.describe()
data_new=data.head()

data_new
#ilk 5 filme bakılarak bütçelerini ve gelirlerini alt alta gösterdik

melted=pd.melt(frame=data_new,id_vars="original_title",value_vars=["budget","revenue"])

melted
#bu değişkenleri sütun olarak yazdık

melted.pivot(index = 'original_title', columns = 'variable',values='value')
#ilk 5 ile son 5 satırdaki verileri alt alta yazdırdık

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)

conc_data_row
#sütun olarak

data1 = data['budget'].head()

data2= data['revenue'].head()

conc_data_col = pd.concat([data1,data2],axis =1) 

conc_data_col
data.dtypes
data['title'] = data['title'].astype('category')

data['revenue'] = data['revenue'].astype('float')
#title ın tipini category e dönüştürdük

#revenue nin tipini de floata çevirdik

data.dtypes
data.info()
#buradaki nan verilerinide göster dropna =False

data["homepage"].value_counts(dropna =False)
data1=data

data1["homepage"].dropna(inplace = True) # data1 den nan olanları drop et 

#-inplace = True çıkardığın sonucları data1 e kaydet
assert 1==1 #hata vermeyince yanlışlık yok
#assert 1==2
assert  data['homepage'].notnull().all() # bir şey döndürmez çünkü nan değerleri düştü
# dictionary ile dataframe oluşturma

country=["spain","france"]

population=["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#yeni sütun oluştuma

df["capital"] = ["madrid","paris"]

df
data1 = data.loc[:,["vote_average","vote_count","budget","revenue"]]

data1.plot()
data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x="vote_count",y = "vote_average")

plt.show()
data1.plot(kind = "hist",y = "vote_count",bins = 50,range= (0,250),normed = True)

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "vote_count",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "vote_count",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
print(type(data["revenue"]))     # series

print(type(data[["revenue"]])) 
boolean = data.revenue > 2000

data[boolean]

first_filter = data.revenue > 200000000

second_filter = data.budget > 75000000

data[first_filter & second_filter]
