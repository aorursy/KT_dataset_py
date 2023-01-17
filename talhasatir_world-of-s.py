# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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
data =pd.read_csv('../input/creditcardfraud/creditcard.csv') 
data.head(10) #dataframeniin ilk 10 satırındaki bilgilere bakıyoruz
data.info() #data hakkında bilgi alıyoruz
data
data.columns
data.corr()
f,ax = plt.subplots(figsize=(18, 18)) #tablonun boyutunu ayarlıyoruz

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()#amacımız =veri stunlarının birbirleri ile olan iliskisine bakmak eger 1 se ve bire yakınsa=dogru orantı
#data colonları arasında baglantıya bakıyoruz.

data.V1.plot(kind = 'line', color = 'g',label = 'V1',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.V2.plot(color = 'r',label = 'V2',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot =V1-V2 yazısını tablonun sag üstüne yazdır

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

# Scatter plot

# X=V1 ,Y=V2

data.plot(kind ='scatter',x='V1',y='V2',alpha=0.5,color='blue')

plt.xlabel('v1')

plt.ylabel('v2')

plt.title('Uzaydan Saygılar')



#histogram

#bins =number of bar in figure = veri de sectigim kolonda verilerin oranları

data.V2.plot(kind ='hist',bins =50,figsize =(12,12))
data.V2.plot(kind ='hist')

plt.clf() #histogramı temizliyoruz =0'lıyoruz

#we cannot see plot due to clf()
#create dictionary

dictionary ={'space':'Hello','Mars':'why'}

print(dictionary.keys())

print(dictionary.values())
dictionary['space']='satürn' #dictionary deki keys'in valuesini değiştirdim

print(dictionary)

dictionary['leptop']='casper' #dictionary'e yeni keys ve deger ekliyorum

print(dictionary)

#del dictionary['Mars'] #'Mars' keysini dictionary den siliyorum

print(dictionary)

print('space' in dictionary) #'space' dictinory içinde varsa true döner yoksa false =bu sadece keys ler için

dictionary.clear() #dictionary'i temizliyorum =içidneki verileri

print(dictionary)
series =data['V1']

print(series)

print('uzay')

dataframe=data[['V1']]

print(dataframe)

print('-------------------')

series2 =data[['V1','V2']]

print(series2)
x=data['V1'] > 2.4 

print(data[x])##v1 kolonundaki degeri = 2.4 den büyük degerli satırları getir
data[np.logical_and(data['V1']>2.3 ,data['V2']>-0.5)] #iki sartı saglayanları yazdır
data[(data['V1']>2.3) &(data['V2'] > -0.3)]
i=0 #1-4 arası sayıları yazdıran while

while i !=5:

    print(i)

    i+=1

lis =[1,2,3,4,5] #dizi elemalarıni yazdiren for

for i in lis:

    print(i)
for index,i in enumerate(lis): #for ile dizi indisi ve karsılık gelen degerin yazımı

    print(index,' : ',i )

    

dictionary={'Türkiye':'Ankara','Almanya':'berlin','Abd':'Washington'}

print(dictionary.keys())

for index,i in dictionary.items():

    print(index,' : ',i)

    