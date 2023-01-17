# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#IT is a  basic kernel I studied on wines datasets 
#I follow the codes and tried to apply on new dataset
#Big Thanks to Kaan Can @kanncaa1 , IF u would like to follow the codes you can find more on here : https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/winemag-data_first150k.csv') #read the data and save as data
data.info()
#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head()
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df=data.groupby(['country','points']).size()
df=df.unstack()
df.plot(kind='bar',figsize = (15,15))
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('countries')              # label = name of label
plt.ylabel('points')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
df=data.groupby(['country']).size()
df.plot(kind = 'hist',bins = 100,figsize = (12,12))
plt.show()
# 1 - Filtering Pandas data frame
tr = data['country']=='Turkey'     # 
data[tr].shape
data[tr]

# 2 - Filtering pandas with logical_and
# There are only 4 wines which are from Turkey and the price is more than 50
data[np.logical_and(data['country']=="Turkey", data['price']>50.0 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['country']=="Turkey" )&( data['price']>50.0)]
data.shape
data.info()
# For example lets look frequency of country
print(data['country'].value_counts(dropna =False))  # if there are nan values that also be counted
data.describe()
# For example: compare points of wines that belongs to same province
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='points',by = 'province')
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = data['country'].head()
data2= data['price'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
data.dtypes
# Lets drop nan values
data_tr=data   # also we will use data to fill missing value so I assign it to data1 variable
print(data_tr.shape)
print(data_tr.dropna().shape)
data_tr.dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
print(data_tr.shape)  # ??
print(data_tr.dropna().shape)


# # With assert statement we can check a lot of thing. For example
assert data.columns[1] == 'country'
assert data.points.dtypes == np.int
# Plotting all data 
data_tr= data.dropna()
data1 = data_tr.loc[:,["points","price","province"]]
data1.plot()
# it is confusing
data1.plot(subplots=True)
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="price",y = "points")
plt.show()
# hist plot  
data1.plot(kind = "hist",y = "price",bins = 50,range= (0,2000))#,normed = True
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "points",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "points",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
filter2trans = data.price > 250
data2trans =data[filter2trans]
data2trans.shape
# Plain python functions
def div(n):
    return n/2
data2trans.price.apply(div)
#lambda function
data2trans.price.apply(lambda n : n/2)
data.set_index(["country","province"])