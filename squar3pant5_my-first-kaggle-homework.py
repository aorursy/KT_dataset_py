# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import codecs







# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Pokemon.csv')
data.info()
data.head()
data.tail()
data.columns
data.shape
print(data['Type 1'].value_counts(dropna =False))
data.describe()
#Örneğin:efsanevi olan veya olmayan pokemon saldırılarını karşılaştırabilir

# siyah çizgi at top is max

# mavi çizgi at top is 75%

# kırmızı çizgi is median (50%)

# mavi çizgi at bottom is 25%

# siyah çizgi at bottom is min

# Aykırılık yok

data.boxplot(column='Attack',by = 'Legendary')

plt.show()
data_new=data.head()

data_new
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['Attack'].head()

data2= data['Defense'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
data.dtypes
data["Type 2"].value_counts(dropna =False)
data1=data

data1["Type 2"].dropna(inplace = True)
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
country=["Spain","France"]

population=["11","12"]

list_label=["country","population"]

list_col=[country,population]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
#Add new columns

df["capital"]=["madrid","paris"]

df
#broadcasting

df["income"]=0

df
#ploting all data

data1=data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

plt.show()
data.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="Attack",y="Defense")

plt.show()
data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True)

plt.show()
fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind='hist',y='Defense',bins=50,range=(0,250),normed=True,ax=axes[0])

data1.plot(kind='hist',y='Defense',bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt.show()
time_list=["1992-03-08","1992-04-12"]

print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
data2=data.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object=pd.to_datetime(date_list)

data2['date']=datetime_object

data2=data2.set_index('date')

data2
print(data2.loc['1993-03-16'])

print(data2.loc['1992-03-10':'1993-03-16'])
data2.resample('A').mean()

data2.resample('M').mean()
data2.resample('M').first().interpolate("linear")