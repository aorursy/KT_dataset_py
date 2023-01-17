# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/merged_5s.csv') # pandas kütüphanesi kullanarak data setimizi okuyoruz
data.info() #data setimiz hakkında genel bilgi veriyor.
data.corr
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(30) # ilk 30 satırı göreceğiz
data.tail(30) #son 30 satırı göreceğiz
data.columns
plt.scatter(data.Flow_IAT_Mean,data.Flow_IAT_Min,color='red',alpha=0.5)#Scatter grafik çizdirme
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Flow_IAT_Mean.plot(kind = 'line', color = 'g',label = 'Source IP',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Flow_Duration.plot(color = 'r',label = 'Destination IP',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.Destination_Port.plot(kind = 'hist', bins = 100)
data.Flow_Duration.plot(kind = 'hist', bins = 100)

data.Source_Port.plot(kind = 'hist', bins = 150)
x = data['Flow_Duration'] > 1000000

y = data['Source_Port'] <10000

data[np.logical_and(x,y)]
data[np.logical_and(data['Destination_Port']>5000,data['Source_Port']<5000)]
for index,value in data[['Source_IP']][0:10].iterrows(): 

    print(index," : ",value)
threshold = sum(data.Flow_Duration)/len(data.Flow_Duration) #Flow duration ortalaması aldık

data["Flow_Duration_Level"] = ["high" if i > threshold else "low" for i in data.Flow_Duration]

data.loc[:10,["Flow_Duration_Level","Flow_Duration"]]

data.shape #data boyutumuzu belirliyoruz
data.describe() # numeric olarak datalarımızın değerlerini verir
# Görsel olarak datayı inceleyeceğiz

data.boxplot(column='Source_Port', by = 'Label')

data.boxplot(column = 'Flow_Duration', by = 'Label')

data.boxplot(column = 'Destination_Port', by = 'Label')

data.boxplot(column = 'Flow_IAT_Max', by = 'Label')

data.boxplot(column = 'Flow_IAT_Mean', by = 'Label')

data.boxplot(column = 'Active_Max', by = 'Label')



data_new = data.head(15) #datamın ilk 20 verisi data new olarak atandı 

data_new
#yeni datamızda sadece belli değerleri almak istiyoruz bu sebeple melted fonksiyonu kullanıyoruz

melted = pd.melt(frame=data_new,id_vars = 'Source_IP', value_vars = ['Source_Port','Destination_Port'])

melted
# melted edilmiş datalarımızı tekrar eski haline getirmek için melted.pivot fonksiyonu kullanılır

# iki data frame i birleştircez

data_1 = data.head(10)

data_2 = data.tail(10)

print(data_1,data_2)
conc_data_row = pd.concat([data_1,data_2],axis=0,ignore_index = True)

conc_data_row # concarete edilmiş yani birleştirilmiş dataların çıktıları

#dataların ilk beş Source_Port ve Labellerini birleştircez

data1 = data['Source_Port'].head(10)

data2 = data['Label'].head(10)

conc_data_col = pd.concat([data1,data2],axis=1)# axis 1 olursa dikey birleştiriyor axis 0 olursas yatay birleştiriyor

conc_data_col
data.dtypes #elimizdeki dataların tipleri
data.info()
assert data['Flow_Bytes'].notnull().all() #Flow bytes datalarında null olmadığını kontrol ettik
assert data.columns[0] == 'Source_IP' #Sutünların ilk elemanın source ıp olduğunu kontrol ettik
fig, axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind = "hist",y = "Source_Port",bins = 50,range= (0,100000),normed = True,ax = axes[0])

data.plot(kind = "hist",y = "Source_Port",bins = 50,range= (0,100000),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 # datalarıma time series ekledik
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])#bu aralıktaki bütün değerleri yazdırcak
data2.resample("A").mean() #yıllara göre datalarımızı sıralıyor
data2.resample("M").mean() #aylara göre datalarımızı veriyor
data2.resample("M").first().interpolate("linear") #aralardaki değerleri lineer olarak dolduruluoyr
data['Destination_Port'][1000]
data.Destination_Port[5454]
data[['Destination_Port', 'Source_Port']]
data.loc[1,["Flow_Duration"]]
data.Source_IP[data.Flow_Duration<150]
data["total_port"] = data.Source_Port + data.Destination_Port

data.head()