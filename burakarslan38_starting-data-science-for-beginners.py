# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.columns
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.PetalWidthCm.plot(kind = "line",color = "r",label="PWidth",linestyle=":",linewidth=2,grid=True)
data.SepalWidthCm.plot(color = "b",label="SWidth",linewidth=2,linestyle="-.",grid=True)
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.show()
#plt.scatter(data.PetalLengthCm,data.SepalLengthCm,alpha=0.5,color = "r")

data.plot(kind = "scatter",x="PetalLengthCm",alpha = 0.5 , color="r",
          y="SepalLengthCm")

plt.show()
data.SepalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
print("Mean of SepalLengthCm: ",(sum(data.SepalLengthCm)/len(data.SepalLengthCm)))
dictionary = {'Turkey' : 'Kayseri','Germany' : 'Münih'}
print(dictionary.keys())
print(dictionary.values())

dictionary["Turkey"]="Bursa" # add a new value
print(dictionary)
print(" ")
dictionary["Germany"] = "München" # update existing value
print(dictionary)
print(" ")
print("New York"in dictionary) # check 
print(" ")
del dictionary["Germany"]# remove "Germany" from dictionary
print(dictionary)
print(" ")
dictionary.clear()#clear all values
print(dictionary)
print(" ")

series = data["SepalLengthCm"] 
print(type(series))
data_frame = data[["SepalLengthCm"]] # İf you dont use double [] you cant use dataframe classes!
print(type(data_frame))
x=data["PetalLengthCm"]>6
data[x]
data[(data['PetalLengthCm']>6) & (data['PetalWidthCm']>2.2)]
for i in dictionary:
   print(dictionary[i])
for key, value in dictionary.items():
    print(key," : ",value)
data.info()
mean_of_sepallength = sum(data.SepalLengthCm) / len(data.SepalLengthCm)
#print(mean_of_sepallength)
data["LeafLength"] = ["long" if i>mean_of_sepallength else "short" for i in data.SepalLengthCm]

print(data[50:65])

print(data['SepalLengthCm'].value_counts(dropna =False))
data.describe()
data.boxplot(column='SepalLengthCm',by = 'SepalWidthCm')
plt.show()
newdata1 = data.head()
newdata2 = data.tail()
conc_data = pd.concat([newdata1,newdata2],axis=0,ignore_index = True )
#ignore_index adds new id column
#if axix=1 it will be concatenate datas horizontly
conc_data
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=conc_data , id_vars="Species",value_vars=["SepalLengthCm","PetalLengthCm"])
melted
data.dtypes
data['SepalLengthCm'] = data['SepalLengthCm'].astype('str')
#We converted SepalLengthCm float to str
data.dtypes
data1=data
data1["Species"].dropna(inplace = True)# we droped nan values in species
data1["Species"]# there is no missing data
country = ["Turkey","Germany"]
population = ["80","50"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df

df["capital"] = ["Ankara","Münih"]
df
data1 = data.loc[:,["SepalLengthCm","PetalLengthCm","SepalWidthCm","SepalWidthCm"]]
data1.plot()
#hard to understand
data1.plot(subplots = True)
plt.show()
#you can easily understand now
data.describe()
#how to do str to datatime

import warnings
warnings.filterwarnings("ignore")
# close warning

data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2 
data2.resample("A").mean()
data2.resample("M").mean()
#we havent got enough value for all months
data2.resample("M").first().interpolate("linear")
#fill NaNs linearly
data
data = data.set_index("Id")
data
data['SepalLengthCm'] = data['SepalLengthCm'].astype('float')
filter1 = data.SepalLengthCm> 6
data[filter1]

data2 =data.SepalWidthCm[filter1]
data2
