# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/master.csv')
products=['Milk', 'Eggs']

amount=['3','5']

list_label=['products','amount']

list_col=[products,amount]

zipped= list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
#we can new column



df['brand']=['Nestle','LDC']

df['storage temp.']=4

df
data.info()
data1=data.loc[:,['suicides_no','population']]

data1.plot()

plt.show()
data1.plot(subplots=True)

plt.show()
#scatter plot

data1.plot(kind='scatter',x='suicides_no',y='population') #makes no sense, just to make example

plt.show()
#histogram



data1.plot(kind='hist',y='population',cumulative=False) #numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)[source]

plt.show()
import warnings

warnings.filterwarnings("ignore")



data2=data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object #adding new column

data2= data2.set_index("date")

data2

print(data2.loc["1993-03-16"])
#or we can list between two dates

print(data2.loc["1992-03-10":"1993-03-16"])
data_samp=data2.resample('M').mean()

print(data_samp)
interpolated = data_samp.interpolate(method='linear') #interpolations assigns new values  within the range of a discrete set of known data points.

print(interpolated)
data3=data.head(8) #first 8 values of dataset
data3['population'][2] #it shuld give 289700 which is second inde according to phyton
data3.sex[2]
data3.loc[2,['suicides_no']] #it can be done by loc function as well
#now we can try to select certain columns

data3[['age','population']]
print(type(data3['age']))
data3.loc[2:4,'age':'population']
data3.loc[4:2:-1,'age':'population']
#Filtering Data Frame



filter1=data.suicides_no<10 

filter2=data.suicides_no>5

filter3=data.population<10000

data[filter1 & filter2 & filter3]
data3.population[data3.age=='25-34 years']
data3.population.apply(lambda n: n/2)
data3['rate']=data3.suicides_no/data3.population

data3
#as it is seen on table there is no index name and it starts from 0



data3.index=range(1,9,1)

data3.index.name='index'

data3

data4=data3.set_index(['age','sex'])

data4
data3
data3.pivot(index="sex",columns = "age",values="suicides_no") # it shows how many suicides happens according to age and sex from data3 above
data5 = data3.set_index(["age","sex"])

data5
data5.unstack(level=1) #sex is moved as upper index.
#also possible to switsch outer and inner position

data6 = data5.swaplevel(0,1)

data6
data7=data3.head(3)

data7
pd.melt(data7,id_vars="age",value_vars=["sex","population"])
data7.groupby("sex").mean()  #max(),min() etc..
data7.groupby("sex")[['suicides_no','population']].mean()  