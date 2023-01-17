# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
veriler = pd.read_csv("/kaggle/input/abalone-dataset/abalone.csv")
veriler.head()
veriler.info()
veriler.corr()
#Visualization library - (Görselleştirme Kütüphaneleri)

import matplotlib.pyplot as plt

import seaborn as sns
#correlation map

f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(veriler.corr(),annot= True,linewidths=.5, fmt=".2f",ax=ax)

plt.show()
veriler.columns
veriler.Diameter.plot(kind = "line")

plt.show()
veriler.hist(figsize=(20,10))

plt.show()
#Scatter Plot

veriler.plot(kind = "scatter",x = 'Whole weight' , y = 'Rings',alpha = 0.5, color = 'red' )

plt.title("Bütün Ağırlık ve Yaş sütunu Scatter plot")

plt.show()
f1 = veriler["Height"]>1

veriler[f1]

#Boyu 1' den büyük olan bir tek Abolone varmış
veriler[(veriler["Rings"]>7) & (veriler['Diameter']<0.2)]

#Yaşı 7'den büyük, Çapı 0.2'den küçü  2 tane Abalone varmış
veriler.shape
veriler.describe()
veriler.boxplot(column="Length",by="Sex",figsize=(13,13))

plt.show()
veriler.boxplot(column="Length",by="Rings",figsize=(13,13))

plt.show()
yeni_veriler = veriler.head(10)

yeni_veriler
melted = pd.melt(frame=yeni_veriler,id_vars="Sex",value_vars=["Whole weight","Rings"])

melted
yeni_veriler2 = veriler.head()
melted2 = pd.melt(frame=yeni_veriler2,id_vars="Rings",value_vars=["Height"])

melted2
#melted.pivot(index="Sex",columns="variable",values="value")

# İki farklı index içerdiği için hata veriyor
dat1 = veriler.head()

dat2 = veriler.tail()



conc_data_row = pd.concat([dat1,dat2],axis=0)

conc_data_row
conc_data_row = pd.concat([dat1,dat2],axis=0,ignore_index= True)

conc_data_row
dat3 = veriler["Length"].head()

dat4 = veriler["Viscera weight"].head()



conc_data_col = pd.concat([dat3,dat4],axis = 1)

conc_data_col
veriler.info()
veriler["Length"] = yeni_veriler["Length"].astype("category")
veriler.info()
assert veriler["Viscera weight"].notnull().all() #Hiçbir şey döndürmez
veriler["Viscera weight"].fillna("empty",inplace = True)
veriler.plot()
veriler.plot(subplots = True,figsize=(12,12))

plt.show()
fig,axes = plt.subplots(nrows=2,ncols=1)

veriler.plot(kind = "hist",y ='Rings',bins = 25,range = (0,30),ax = axes[0])

veriler.plot(kind = "hist",y ="Rings",bins = 50,range = (0,30),ax=axes[1],cumulative = True)

plt.show()
veriler.Rings.describe()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16","1993-04-15","1993-04-16"]

date_time_object = pd.to_datetime(date_list)
# for close warning

import warnings

warnings.filterwarnings("ignore")



data_time = veriler.head(7)

data_time["date"] = date_time_object
data_time
data_time = data_time.set_index("date")

data_time
data_time.loc["1993-03-16"]
data_time.loc["1993-03-16":"1993-04-16"]
data_time.resample("A").mean()
data_time.resample("A").mean()
data_time.resample("M").mean()
data_time.resample("M").mean().interpolate("linear")
veriler.head()
veriler.set_index("Sex")
veriler["Rings"][3] #Rings sütununun 3. index'teki değeri
veriler.loc[1:10,"Viscera weight":]
veriler.loc[10:0:-1,"Viscera weight":]
veriler.loc[10:0:-1,"Viscera weight":]
veriler.head()
boolean = veriler["Rings"]>7

veriler[boolean]
first_filter = veriler.Rings < 8

second_filter = veriler["Whole weight"] > 1.15 



veriler[first_filter & second_filter]
veriler["Whole weight"].describe()
veriler.Height.apply(lambda x : x/2 )
veriler["Total"] = veriler['Whole weight'] + veriler.Height
veriler.head()
print(veriler.index.name)
veriler.index.name = "index_name"

veriler.head()
datacopy = veriler.copy() #DataFrame'imizi kopyaladık
datacopy.index = range(1000,5177,1)

datacopy.head()
datacopy = datacopy.set_index(["Diameter","Rings"])

datacopy.head()

datacopy2 = datacopy.swaplevel(0,1)

datacopy2.head()
#datacopy.unstack(level = 1)

#Indexlerden "Diameter" olanı kaldırmak istiyorum fakat hata ile karşılaşıyorum, geri dönüş yaparsanız seviirim.
datacopy2 = datacopy2.reset_index()

datacopy2.head()
veriler.groupby("Sex").mean()
veriler.groupby("Sex").Rings.max()
veriler.groupby("Sex")[["Height","Rings"]].min()