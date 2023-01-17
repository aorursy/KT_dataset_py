# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.head()
data.info()
data.SepalLengthCm.plot(kind = 'line', color = 'g', label = 'SepalLengthCm',linewidth=2,alpha=1,grid=True,linestyle=':')

data.PetalLengthCm.plot(kind = 'line', color = 'b', label = 'PepalLengthCm',linewidth=2,alpha=1,grid=True,linestyle = '-.')

plt.legend(loc='lower right')

plt.xlabel= 'SepalLengthCm'

plt.ylabel= 'PetalLengthCm'

plt.title('Line Plot')

plt.show()
# scatter plot

# x = SepalLenghtCm  y = PetalLenghtCm



data.plot(kind = 'scatter', x='SepalLengthCm', y='PetalLengthCm', alpha=1, color='purple')

plt.title('SepalLengthCm - PetalLengthCm Scatter Plot') # title of scatter plot
data.SepalLengthCm.plot(kind ='hist',bins=50,figsize=(12,12))

plt.show()
# dictionary 

dictionary = {'Kocaeli': '41','İzmir':'35','Bursa':'16','Ankara':'06'}

print(dictionary.keys())

print(dictionary.values())
dictionary ['Adana'] = '01'  # insert item to dictionary

print(dictionary)

del dictionary['Adana']      # delete item from dictionary

print(dictionary)

print('Adana' in dictionary)

dictionary.clear()  # delete all items of dictionary

print(dictionary)

del (dictionary)  # delete dictionary
series = data['SepalLengthCm']

print(type(series))

data_frame = data[['PetalLengthCm']]

print(type(data_frame))
# Filtering Pandas data frame

x = data['PetalLengthCm']>6.5

data[x]
data[np.logical_and(data['SepalLengthCm']>7,data['PetalLengthCm']>6.5)]
data[(data['SepalLengthCm']>7) & (data['PetalLengthCm']>6.5)]
list = [29,45,78,96,45,12,0]

for i in list:

    print("i is ",i)

    

for index,value in enumerate(list):

    print(index,":",value)

print('')



dictionary = {'ankara':'06','izmir':'35','bursa':'16','çanakkale':'14'}



for index,value in data[['SepalLengthCm']][0:4].iterrows():

    print(index,":",value)
# LIST COMPREHENSION



average = sum(data['PetalLengthCm'])/ len(data['PetalLengthCm'])

print(average)

data['Petal_level'] = ['high' if i > average else 'low' for i in data.PetalLengthCm]

data.loc[:5,['Petal_level','PetalLengthCm']]
data.head()
 # anonymous function

 # first 5 item in dataframe, SepalLengthCm column

'''length_list = data['SepalLengthCm'][0:5]

print(length_list)

y = map(lambda x:x**2,length_list)

print(list(y))'''
# zip()



"""list1 = []

list.extend(data['SepalLengthCm'][0:5])

list2 = []

list2.extend(data['PetalLengthCm'][0:5])

z = zip(list1,list1)

z_list = []

z_list.extend(list(z))

print(z_list)"""
data.shape
data.info()
data.describe()
data.boxplot(column = 'SepalWidthCm' , grid = False)

plt.show()
# tidy data 

data_new = data.tail(7)

data_new
# melt 

melted = pd.melt(frame = data_new, id_vars = 'Id' , value_vars = ['SepalLengthCm','SepalWidthCm'])

melted
# pivoting : reverse of melt function

melted.pivot(index = 'Id',columns = 'variable' , values = 'value')
data1 = data.head()

data2 = data.tail()

conc_data_raw = pd.concat([data1,data2],axis = 0 ,ignore_index = True)

conc_data_raw
data1 = data['SepalLengthCm'].head()

data2 = data['SepalWidthCm'].head()

conc_data_col = pd.concat([data1,data2],axis=1)

conc_data_col
data.dtypes
data['PetalWidthCm'] = data['PetalWidthCm'].astype('object')
data.dtypes

data['PetalWidthCm'] = data['PetalWidthCm'].astype('float')

data.dtypes
# we can think value_counts function like group by in Sql

data["Petal_level"].value_counts(dropna =False)
data1 = data.head(8)

data1['Petal_level'].dropna(inplace = True)
# data frames from dictionary

list1 = data['PetalLengthCm'][:5]

list2 = data['PetalWidthCm'][:5]

list_label = ["PLength","PWidth"]

list_col = [list1,list2]

zipped = zip(list_label,list_col)

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# add new columns

df['PLevel'] = data['Petal_level']

df

#broadcasting

df['SLevel'] = '0'

df
# Plotting all data 

data1 = data.loc[:,["PetalLengthCm","SepalWidthCm","PetalWidthCm"]]

data1.plot()
# subplots 

data1.plot(subplots = True)

plt.show()
data1.plot(kind='scatter' ,x = 'PetalLengthCm', y = 'PetalWidthCm', color='r')

plt.show()
data1.plot(kind = 'hist' , y = 'PetalLengthCm' , bins=40, range = (0,5), normed= True)

plt.show()
data1.plot(kind = 'hist' , x = 'PetalLengthCm' , bins=40, range = (0,5), normed=True)

plt.show()
data1.plot(kind = 'hist', bins=40, range = (0,5), normed=True)

plt.show()
# histogram subplot with non cumulative and cumulative



fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "PetalLengthCm",bins = 50,range= (0,5),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "PetalLengthCm",bins = 50,range= (0,5),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
# time series

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # str

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2

data2["date"] = datetime_object

data2 = data2.set_index("date")

data2

data2.resample('M').mean()
data2.resample("M").first().interpolate('linear')
data2.resample('M').mean().interpolate("linear")
data = pd.read_csv('../input/Iris.csv')

data.head()

data = data.set_index("Id")

data.head()
# İf didn'n use set_index function , our list gonna be start index from zero.

data["SepalLengthCm"][1]
data.SepalLengthCm[1]

# using loc , 1 -> th row

#             [" bla bla "] column

data.loc[1,["SepalLengthCm"]]
# from dataset , 2 column and theirs 5 entry

data[["SepalLengthCm","SepalWidthCm"]][:5]
# Slicing data frame

print(type(data["PetalLengthCm"]))     # series

print(type(data[["PetalLengthCm"]]))   #data frames

# Slicing and indexing series

data.loc[1:7,"SepalLengthCm":"PetalLengthCm"]
# reverse slicing

data.loc[10:1:-1,"SepalLengthCm":"PetalLengthCm"]
data.loc[145:,"PetalWidthCm":]
# filtering data frames

boolean = data.PetalWidthCm >2.4

data[boolean]
# combining filters

# intersection is null

first_filter = data.SepalLengthCm > 6.5

second_filter = data.SepalLengthCm < 2.0

data[first_filter & second_filter]

first_filter = data.PetalLengthCm > 6.5

second_filter = data.SepalLengthCm > 2.0

data[first_filter & second_filter]

data[data.PetalLengthCm >6.7]
# Filtering column based others

data.SepalLengthCm[data.PetalLengthCm >6.7]
# transforming data

def mult(n):

    return n*2

data.PetalLengthCm.apply(mult)
# lambda function

data.PetalLengthCm.apply(lambda n : n*2)
# defining new coloumn using other coloumns

data['TotalLength'] = data.PetalLengthCm + data.SepalLengthCm

data1 =data.loc[0:5,'SepalLengthCm':'PetalLengthCm']

data1['TotalLengthCm'] = data['TotalLength']

data1
# changing index_name

print(data.index.name)

data.index.name = "index_name"

data.head()
data = pd.read_csv('../input/Iris.csv')

data.head()
data.set_index(['SepalLengthCm','SepalWidthCm'])
# creating dataframe from dictionary

dic = {"Cinsiyet":["Kız","Erkek","Kız","Kız"],"Gozluk":["Evet","Hayır","Evet","Hayır"],

       "Uzunboy":["Evet","Hayır","Evet","Evet"]}

df = pd.DataFrame(dic)

df