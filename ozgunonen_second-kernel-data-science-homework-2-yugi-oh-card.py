# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import seaborn as sns

import matplotlib.pyplot as plt 
data=pd.read_csv("../input/card_data.csv")
data.info()
data.shape
data
data.columns 
print(data["Name"].value_counts(dropna=False))
data.boxplot(column="ATK",by="DEF")
data.boxplot(column="ATK",by="Level")
melting=pd.melt(frame=data,id_vars="Name",value_vars=["ATK","DEF"])

melting
data1 = data['ATK'].head()

data2= data['DEF'].head()

conc_data= pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =1,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data.dtypes
data.Race=data.Race.astype("category")
data.ATK=data.ATK.astype("int")
assert 1==1
assert  data['Race'].notnull().all()
data1=data

data1["ATK"].dropna(inplace=True)

#nan olanları drop etmek için kullanılır   nan olanları listeden at ve çıkardığın sonuçları data1 e at demektir
assert 1==1 #hiçbirşey döndürmediğine göre doğrudur.
data["ATK"].fillna("Boş",inplace=True)
assert data["ATK"].notnull().all()  #nan olanlar listeden atıldığı için hiçbirşey döndürmeyecek 
assert data["ATK"].notnull().all()
data.head(10)
assert data.columns[1] == "Type"
data.head()

data.ATK[1]

data.DEF[4]
data.loc[4,["Level"]]
data[["ATK","Level"]]
#slicing data frames

data.loc[0:5,"Level":"ATK"]
data.loc[10:5:-1,"Level":"DEF"]
data.loc[0:7,"ATK":]
#FİLTERİNG DATA frames

boolean=data.ATK>4500

data[boolean]
data[data.DEF>4410]
data.Type[(data.ATK>4500 )& (data.DEF>4500)]
data[(data.ATK>4500 )& (data.DEF>4500)]
data.Level[(data.ATK>4500 )& (data.DEF>4500)]
def bjk(n):

    return n/2

data.ATK.apply(bjk)
data.ATK.apply(lambda n:n/2) # same things 
data["power"]=data.ATK+data.DEF
data.head()
city=["istanbul","aydın"]

plaka=["34","9"]

list_label=["city","plaka"]

list_columns=[city,plaka]

zipped=list(zip(list_label,list_columns))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["bolge"]={"marmara","ege"}
df
df["turkey"]=1
df
data3=data.loc[:,["Level","ATK","DEF"]]

data3.plot()
data3.plot(subplots=True)
data3.plot(kind="hist",y="ATK",bins=50,cumulative=True,normed=True,range=(0,6000))


data2=data.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

dateobject=pd.to_datetime(date_list)

data2["date"]=dateobject

data2=data2.set_index("date")

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")