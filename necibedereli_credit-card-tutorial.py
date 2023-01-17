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
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv') #tabloyu cagirdik

data.head()
data.info() #bilgilerini inceleyecegiz
data.corr()
data.columns #sütunlarini görecegiz
country=["Spain","France"]

population=["11","12"]

list_label=["country","population"]

list_col=[country,population]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["capital"]=["Madrid","Paris"] #yeni bir column ekledik

df
df["income"]=0 #broadcast ekledik. Aslında yukarıdaki kod ile aynı şey.

df
data1=data.loc[:,["Time","V1","V2"]] #credit cardın pllot özelligi

data1.plot()

data1.plot(subplots=True) #degerleri ayri ayri göstersin karismasin istersek subplot

data1.plot()

data1.plot(kind="scatter", x="Time", y="V1") #scatter plot şeklinde görüntülememizi sagladi

data1.plot()

data1.plot(kind="hist",y="Time", normed=True)
fig, axes = data1.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Time",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Time",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

data1.savefig('graph.png')

data1
data.describe()
time_list=["1992-03-08","1992-04-12"]

print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
import warnings #hata özelligini kapatmak icin

warnings.filterwarnings("ignore")

data2=data.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object=pd.to_datetime(date_list)

data2["date"]=datetime_object

data2=data2.set_index("date")

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean() #yila göre columnsların ortalama degerleri
data2.resample("M").mean() #aya göre columnsların ortalama degerleri
data2.resample("M").first().interpolate("linear") #naN degerlerinin icerigini ortalama alarak doldurdu
data2.resample("M").mean().interpolate("linear")