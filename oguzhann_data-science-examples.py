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
data= pd.read_csv("../input/Iris.csv")

data.head()
data.info()
data.corr()
data.columns
data.SepalLengthCm.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=2,alpha = 0.5,grid = True,linestyle = '-.')

data.PetalLengthCm.plot(color = 'r',label = 'Defense',linewidth=2, alpha = 0.5,grid = True,linestyle = '-')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.SepalLengthCm.plot(kind = 'hist',bins = 70,figsize = (15,6))

plt.show()
# 1 - Filtering Pandas data frame

x = data['SepalLengthCm']>7     # There are only 3 pokemons who have higher defense value than 200

data[x]
data[(data['SepalLengthCm']>7) & (data['SepalWidthCm']>2)]
data[np.logical_and(data['SepalLengthCm']>7 ,data['SepalWidthCm']>2)]
# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Id', value_vars= ['PetalWidthCm','Species'])

melted
# Firstly lets create 2 data frame

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['PetalLengthCm'].head()

data2= data['SepalLengthCm'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
data['SepalLengthCm'] = data['SepalLengthCm'].astype('int64')

data['PetalWidthCm'] = data['PetalWidthCm'].astype('int64')
data.dtypes
# Lets chech Type 2

data["PetalWidthCm"].value_counts(dropna =False)
# Plotting all data 

data1 = data.loc[:,["PetalWidthCm","SepalLengthCm","SepalWidthCm"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
data.describe()
# Creating boolean series

boolean = data.SepalWidthCm > 4

data[boolean]
first_filter = data.SepalWidthCm > 3

second_filter = data.PetalLengthCm > 3

data[first_filter & second_filter]
# read data

datax = pd.read_csv('../input/Iris.csv')

datax.head()
datax.loc[1,["SepalWidthCm"]]
datax[["SepalLengthCm","SepalWidthCm"]]