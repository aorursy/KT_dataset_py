# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.info()
data.head(10)
#en yüksek korelasyon ilişkisi oldpeak ile slope arasında görülmüştür. Genel yorum yapacak olursak değişkenler arasında ciddi bir ilişki tespit edilmemiştir.

data.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
#her bir hasta için yaşları ve kolestrol değerleri görülmektedir. Göze çarpan kolestrol değerlerinde bir uç değer olduğudur. Ortalama kolestrol değeri grafiğe bakıldığında 2570 civarı olarak görülmekte ve 400 üzeri için çok gözlem yoktur..



# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.age.plot(kind = 'line', color = 'g',label = 'age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.chol.plot(color = 'r',label = 'chol',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='age', y='chol',alpha = 0.5,color = 'blue')

plt.xlabel('age')              # label = name of label

plt.ylabel('chol')

plt.title('age chol Scatter Plot')            # title = title of plot



#scatter plotumuz da korelasyon matrisimiz ile benzer şekilde sonuç vermiştir. Aralarındaki zayıf ilişiki olduğu görülmektedir.
data.age.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()



#datamızda kitlenin yaş dağılımımı görebiliyoruz.
series = data['trestbps']        # data['trestbps'] = series

print(type(series))

data_frame = data[['age']]  # data[['age']] = data frame

print(type(data_frame))
x = data['age']<30

data[x]

#30 yaşından küçük olan tek değer tespiti
y=data["chol"]>500

data[y]

#kolestrol için uç değer tespiti


data[np.logical_and(data['age']<45, data['chol']<200)]

#45 yaşından küçük kolestrol 200 den küçük kitlenin değerleri tespit edilmiştir.
# kitlemizdeki kolestrol değeri yüksek olanlar "high risk" diğerleri "normal" olarak etikelenmiştir.



threshold = sum(data.chol)/len(data.chol)

print(threshold)

data["chol_level"] = ["high risk" if i > threshold else "normal" for i in data.chol]

data.loc[:10,["age","chol","chol_level"]]



# ders3 buradan başlamakta





data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head(10)
data.tail()
data.columns #sütun isimleri kontrol edildi



data.shape #satır sütun sayısı kontrol edildi
data.info() #veri setinde null gözlem bulunmamakta
print(data.age.value_counts(dropna =False)) #yaş gruplarından kaçar adet gözlem olduğu kontrol edildi null gözlem olmasa da var ise sayılması istendi



data.describe() #genel veri tanımlamaları
data.boxplot(column='age', by = 'sex') #cinsiyete göre yaşlarda ıç gözlem bulunmamaktadır

data_new = data.head()    # ilk data alındı

data_new


melted= pd.melt(frame=data_new,id_vars ='age',value_vars=['chol','thalach'])

melted
melted.pivot(index = 'age', columns = 'variable',values='value')


data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis=0 ,ignore_index = False) 

conc_data_row #indexler korunarak dikey birleştirme yapıldı




data1= data['chol'].head()

data2= data['trestbps'].head()

conc_data_col = pd.concat([data1,data2],axis=1)

conc_data_col
data.dtypes
data['trestbps'] = data['trestbps'].astype('float64') #int to float

data['oldpeak'] = data['oldpeak'].astype('int64') #float to int

data.dtypes
data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.info()


data["target"].value_counts(dropna=False)
data1=data

data1["age"].dropna(inplace = True) #boş değerleri sil
assert  data['age'].notnull().all()
 #assert  data['age'].null().all() # age sütununda nulldeğer var mı?
#ders4 pandas



data1=data.loc[:,["age","chol","trestbps"]]

data1.plot()


data1.plot(subplots =True)

plt.show()


data1.plot(kind="scatter",x="age",y="chol")

plt.show()
#chol değerlerinin kümülatif ve kümülatif olmayan frekanslarının grafikleri ile hangi yaş değerinde yoğunlaşma olduğu tespiti amaçlandı

fig, axes = plt.subplots(nrows=2,ncols=1) 

data1.plot(kind = "hist",y = "chol",bins = 50,range= (100,450),normed = False,ax = axes[0])

data1.plot(kind = "hist",y = "chol",bins = 50,range= (100,450),normed = False,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()

#veri hakkında genel bilgi edinme amaçlı değerler getirildi
data2 = data.head() #tarih listesi oluşturuldu. liste önce zaman serisine sonra indexe dönüştürüldü

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2
print(data2.loc["1992-03-10":"1993-03-16"])

#tarihler arasındaki gözlemler kontorl edildi
data2.resample("A").mean() # yıl bazlı ortalamalar


data2.resample("M").mean().interpolate("linear")  #boş değer olmadığında  sadece ortalamalar alındı
#ders5

data = pd.read_csv('../input/heart-disease-uci/heart.csv')
#ilk 5 değer için yeniden index oluşturuldu

data_index=[1,2,3,4,5]

data2=data.head()

data2["index"]=data_index

data2 = data2.set_index("index")

data2
data2.chol[1] #ilk chol değeri kontrol edildi
data2.loc[1,["chol"]]


print(type(data2["chol"]))

print(type(data2[["chol"]]))
#belirli alanı gözleme

data2.loc[2:,"thalach":,]
data2.loc[2:4,"chol"]

#kriterlere uygun değerleri alt alta getirme

first_filter = data2.chol > 249

second_filter = data2.thalach > 150

data2[first_filter & second_filter]


def div(n):

    return n/2

data2.chol.apply(div)
#toplam değer hesabı

data2["total_value"] = data2.chol + data2.thalach

data2.head()
data3 = data.copy()

data3.head()





 #data3.index = range(100,900,1)

 #data3.head()
#baştan index oluşturma

data['index_new'] = range(1, len(data) + 1)



data = data.set_index("index_new")

data.head()

data.tail()
data3 = data.copy()

data3.head()



#idex atama

data3.index = range(50,353,1)

data3.head()
# dataframe oluşturma

data4=data.head().copy()

df=pd.DataFrame(data4)

df
df.pivot(index="sex",columns = "age",values="oldpeak")

#cinsiyetler bazında ortalma değerler

df.groupby("sex").mean()
#cinsiyetlere göre max chol değeri

df.groupby("sex").chol.max()