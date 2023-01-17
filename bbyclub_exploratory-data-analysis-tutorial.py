# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/data.csv")
data.info()
data.head()
data.columns
#count and groupby the nationality column

print(data["Nationality"].value_counts(dropna=False))
data_new= data.head()

data_new
melted=pd.melt(frame=data_new, id_vars="Name", value_vars=["Overall","Potential"])

melted
#Reverse melting

melted.pivot(index="Name", columns="variable", values="value")
data1=data.head()

data2=data.tail()

concat_data_row= pd.concat([data1,data2], axis=0, ignore_index=True) 

#by axis=0 add data frames one after other

#ignore_index : create new index for rows

concat_data_row
data1=data["Name"].head()

data2=data["Nationality"].head()

concat_data_col= pd.concat([data1,data2], axis=1)

#by axis=0 add data frames one after other

#ignore_index : create new index for rows

concat_data_col
data.dtypes
#Convert column type

data["Overall"]= data["Overall"].astype("float", error_bad_lines=False)
data.dtypes
#data frames from dictionary

country=["Turkey", "Greece", "Russia"]

population=["70","80","100"]

list_label=["country","population"]

list_col=[country,population]

zipped=list(zip(list_label,list_col)) 

data_dict=dict(zipped) #convert to dictionary

df=pd.DataFrame(data_dict)

df
#add new column = Brodcasting: Create new column and assign different value to entire column

df["capital"]=["Ankara","Athens","Moscow"]

df

df["income"]=0 #Boradcasting entire column

df
#plotting all data

data1=data.loc[:,["Strength","LongShots","Aggression"]]

data1.plot()
#divide into subplots

data1.head(100).plot(subplots=True)

plt.show()
data1.head(100).plot(kind="scatter", x="Strength", y="Aggression")

plt.show()
data1.plot(kind="hist", y="LongShots",bins=50,range=(0,250),normed=True) #normed normalize etmek
#histogram subplot with cumulative and non-cumulative

fig,axes= plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist", y="Strength", bins=50, range=(0,250), normed=True, ax=axes[0])

data1.plot(kind="hist", y="Strength", bins=50, range=(0,250), normed=True, ax=axes[1], cumulative=True)

plt.savefig("graph.png")

plt
#Pandas Time Series

time_list=["1992-03-08", "1992-04-12"]

print(type(time_list[1])) #we want to convert string to daettime object

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")



data2=data.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object= pd.to_datetime(date_list)

data2["date"]=datetime_object

#Lets make date as index

data2=data2.set_index("date")

data2

#Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1993-03-10":"1993-03-16"])
data.head()
print(data.index.name)
data.index.name="index_name"

data.head()
data=data.drop(["Unnamed: 0"],axis=1)
data.head()

data3= data.copy() #deep copy

data3.index=range(100,18307,1) #index start from 100. It's arbitrary value

data3.index.name="index_name"

data3.head()
data4=data3.set_index("ID")
x=data[data['Name'].str.contains("Ibrahim")] #search a substring

x
data5=data.set_index(["Club","Nationality"])

data5.head(100)
s=(data.groupby("Nationality").groups)

grouped = data.groupby('Nationality')

print(type(grouped))

#for name,group in grouped:

#    print(name)

#    print(group)

grouped.get_group('Turkey') 
s
#group by nationality and sort by potential

gr=data.sort_values(['Potential'],ascending=False).groupby('Nationality').head(500)

gr = gr.groupby('Nationality')

gr.get_group('Turkey')
data.pivot(index='ID', columns="Nationality", values="Potential")
data
data.reset_index()
age_group=data.groupby("Club")

age_group.get_group("FC Barcelona")
data.reset_index()

data.reset_index()

age_group=data.groupby("Club")

print(age_group)

age_group.get_group("FC Barcelona").Age.mean()
data.info()
data[data.Age>38]
data=data.reset_index()
#Important. put na as 2nd parameter to ignore Nan

data[data['Club'].str.contains("Galatasaray",na=False)]