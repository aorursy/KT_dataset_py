
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input")) # we can import data from this directory
# first we need to import data which we use
mydata = pd.read_csv('../input/column_2C_weka.csv')
# data info (name of columns, data type, memory usage)
mydata.info()
#first seven datas
mydata.head(7)
# last 7 datas
mydata.tail(7)
#name of the columns
mydata.columns
# command shape gives number of rows and columns in a tuble
mydata.shape
#For example lets look frequency of our data
print(mydata["class"].value_counts(dropna =False))  
# we can find mean,std... values by the following command
mydata.describe() #ignore null entries
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
mydata.boxplot(column="pelvic_incidence",by = "pelvic_radius")
# Firstly I create new data from our data to explain melt more easily.
mydata_new = mydata.head()    # I only take 5 rows into new data
mydata_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=mydata_new,id_vars = "pelvic_incidence", value_vars= ["sacral_slope","pelvic_radius"])
melted
melted.pivot(index = "pelvic_incidence", columns = 'variable',values='value')
# First we need 2 data frame
data1 = mydata.head()
data2= mydata.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = mydata["sacral_slope"].head()
data2= mydata["pelvic_radius"].head()
conc_data_col = pd.concat([data1,data2],axis =1)
# axis = 0 : adds dataframes in row
conc_data_col

mydata.dtypes
# lets convert float to categorical
mydata["sacral_slope"] = mydata["sacral_slope"].astype('category')

mydata.dtypes
# Lets chech pelvic_radius
mydata["pelvic_radius"].value_counts(dropna =False)
# As you can see, there is not null values in our datas