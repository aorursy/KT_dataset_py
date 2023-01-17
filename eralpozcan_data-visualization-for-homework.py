# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.info()
df.head() #Sayı girmezsek ilk 5 veriyi görürürüz.
df.isnull().sum()
#Farklı bir kernelde host_id - host_name ve last_review'in gereksiz olduğunu ve etik olmayan bilgiler içerdiği için silmeyi tercih etmiş.

#Last review zaten 10k gibi bir değeri sıfır. 

df.drop(['id','host_name','last_review'],axis=1,inplace=True)

df.head(5)
#Reviews_per_month değerlerini olmayanları NaN olanları sıfır ile değiştiriyoruz

df.fillna({'reviews_per_month':0},inplace=True)

df.reviews_per_month.isnull().sum()
#NaN değerlerimizi 0 yaptığımızı görebiliyoruz. Verimiz şu an temiz bir durumda ilk duruma göre.

df.head()
#Oda Tiplerini görüntüledik.

df.room_type.unique()
df.neighbourhood_group.unique()

#5 Bölge mevcut.
#Burada en çok hangi odanın tercih edildiğini sorguluyoruz. 

top_room_type = df.room_type.value_counts().head(5)

top_room_type
sns.set(rc={'figure.figsize':(10,8)})
##Doğru şekil çizdirmeyi seçmenin önemini bu satırda anlıyorum 

top_room_type_viz = top_room_type.plot(kind ='hist')

top_room_type_viz.set_title('Best Room Type')

top_room_type_viz.set_ylabel('Count')

top_room_type_viz.set_xlabel('Best Type')

top_room_type_viz.set_xticklabels(top_room_type_viz.get_xticklabels(), rotation=45)
#

roomtypevis = top_room_type.plot(kind ='bar')

roomtypevis.set_title('Best Room Type')

roomtypevis.set_ylabel('Count')

roomtypevis.set_xlabel('Best Type')

roomtypevis.set_xticklabels(roomtypevis.get_xticklabels(), rotation=45)
describe_f=df.price

describe_f.describe()
ortalama=np.average(df.price)

ortalama

#Tüm şehirin fiyat ortalaması 152$
#df[np.logical_and(df['room_type']=='Shared room', df['price']<500 )]
#sharedroom =df[df.room_type == 'Shared room']

sharedroom = df[np.logical_and(df['room_type']=='Shared room', df['price']<500 )]

sharedroom_viz = sns.violinplot(data=sharedroom, x='room_type',y='price')

sharedroom_viz.set_title('Shared Room / Price')
price_room_type = df[df.price<600]

price_room_type = sns.violinplot(data=price_room_type,x='room_type', y='price')

price_room_type.set_title('Ev / Oda Tipine Göre Ücretler ')
#Burada normalde 1000 seçmiştim bir çok veriyi görmek için ama kötü çıktı farklı kernelde bunu 500 gibi bir değer almışlar daha güzel görmek için.Hem bakıldığında neredeyse manhatın dışında +400 civarlarında aşırı derecede az yer var.

#Ben 600 olarak seçiyorum.

price_Vi = df[df.price<600]

price_Viz = sns.violinplot(data=price_Vi,x='neighbourhood_group', y='price')

price_Viz.set_title('Bölgelerin Ücretleri')

manhattan_f_average=df[np.logical_and(df['neighbourhood_group']=='Manhattan', df['price']<700 )]

manhattan_average=np.average(manhattan_f_average.price)

print("Manhattan Average(Ortalama Fiyat) :",manhattan_average)



brooklyn_f_average=df[np.logical_and(df['neighbourhood_group']=='Brooklyn', df['price']<700 )]

brooklyn_average=np.average(brooklyn_f_average.price)

print("Brooklyn Average :",brooklyn_average)



bronx_f_average=df[np.logical_and(df['neighbourhood_group']=='Bronx', df['price']<700 )]

bronx_average=np.average(bronx_f_average.price)

print("Bronx Average:",bronx_average)

df.plot(kind='scatter',color='r',x='longitude',y='latitude',label='availability_365',alpha=0.5,grid=True)

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title('Longtitude-Latitude')

plt.show()
#Seaborn scatterplot example.

sns.set(style="darkgrid") ##Arkaya ızgara atıyor.





f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.despine(f, left=True, bottom=True)

clarity_ranking = ["longitude", "latitude"]

sns.scatterplot(x="longitude", y="latitude",

                palette="ch:r=-.2,d=.3_r",

                hue_order=clarity_ranking,

                sizes=(1, 2), linewidth=0,

                data=df, ax=ax)