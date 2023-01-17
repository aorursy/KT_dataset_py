# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')



data.info()
data.corr()

# it seems that there is a correlation between SepalLength and PetalLength
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)

data.columns
# Ploting Features



# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.SepalLengthCm.plot(kind = 'line', color = 'g',label = 'SepalLengthCm',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.PetalLengthCm.plot(color = 'r',label = 'PetalLengthCm',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='SepalLengthCm', y='PetalLengthCm',alpha = 0.5,color = 'red')

plt.xlabel('SepalLengthCm')              # label = name of label

plt.ylabel('PetalLengthCm')

plt.title('SepalLengthCm - PetalLengthCm Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.SepalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# Pandas Lib. Studies



series = data["SepalLengthCm"]

print(type(series))

data_frame = data[["SepalLengthCm"]]

print(type(data_frame))
# Filtering the data

x = data["SepalLengthCm"] > 7

data[x]

# there are 12 instances that has more than 7 cm sepal length



# Filtering pandas with logical and&



data[(data["PetalLengthCm"]>6) & (data["SepalLengthCm"] > 6)]



#there are 9 subjects that satisfy the condition





threshold = sum(data.SepalLengthCm)/len(data.SepalLengthCm)

data["sepal_level"] = ["high" if i > threshold else "low" for i in data.SepalLengthCm]

data.loc[:50,["sepal_level","SepalLengthCm"]]  

# In first 50 samples there is only one subject that is above our theshold
data.head()

data.tail()
# exploratory data analysis



print(data['Species'].value_counts(dropna =False)) 

# with value_counts method we can obtain frequency or amount of subjects. 

data.describe() #ignore null entries
#VISUAL EXPLORATORY DATA ANALYSIS

#Box plots: visualize basic statistics like outliers, min/max or quantiles



# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='SepalWidthCm',by = 'sepal_level')
# Tidy Data



data_new = data.tail()

data_new
# melting

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Species', value_vars= ['PetalLengthCm','PetalWidthCm'])

melted
#PIVOTING DATA

#Reverse of melting.



#melted.pivot(index = 'Species', columns = 'variable',values='value')
# CONCATENATING DATA

#We can concatenate two dataframe



# Firstly lets create 2 data frame

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row



#vertical data conc.
data1 = data['SepalWidthCm'].head()

data2= data['SepalLengthCm'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col



#horizontal data conc.
data.dtypes
# lets convert object(str) to categorical and int to float.

data['Species'] = data['Species'].astype('category')

data['Id'] = data['Id'].astype('float')
# As you can see Species is converted from object to categorical

# And Id converted from int to float

data.dtypes
data.info()

data1=pd.read_csv('../input/Iris.csv')

data1["Species"].value_counts(dropna =False)

data1.info()
# data frames from dictionary

country = ["Iceland","Sweden"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

list_col

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"] = ["Reykjavik","Stockholm"]

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column

df
data1 = data.loc[:,["SepalWidthCm","SepalLengthCm","PetalLengthCm"]]

data1.plot()
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="SepalWidthCm",y = "SepalLengthCm")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "PetalLengthCm",bins = 50,range= (0,250),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "SepalLengthCm",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "SepalLengthCm",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()

#describe method to see statistical exploratory data analysis
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])


#RESAMPLING PANDAS TIME SERIES

# We will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
# read data

data = pd.read_csv('../input/Iris.csv')

data= data.set_index("#")

data.head()
# indexing using square brackets

data["PetalWidthCm"][1]
# using column attribute and row label

data.PetalWidthCm[1]
# using loc accessor

data.loc[1,["PetalWidthCm"]]
# Selecting only some columns

data[["PetalLengthCm","PetalWidthCm"]]
# Difference between selecting columns: series and dataframes

print(type(data["PetalLengthCm"]))     # series

print(type(data[["PetalLengthCm"]]))   # data frames



# Slicing and indexing series

data.loc[1:10,"SepalLengthCm":"PetalLengthCm"]   # 10 and "PetalLengthCm" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"SepalLengthCm":"PetalLengthCm"] 
# From something to end

data.loc[1:10,"SepalLengthCm":] 
#FILTERING DATA FRAMES



#Creating boolean series Combining filters Filtering column based others



# Creating boolean series

boolean = data.SepalLengthCm > 7

data[boolean]
# Combining filters

first_filter = data.SepalLengthCm > 7

second_filter = data.SepalWidthCm > 3.2

data[first_filter & second_filter]
#TRANSFORMING DATA



# Plain python functions

def div(n):

    return n/2

data.SepalWidthCm.apply(div)
# Or we can use lambda function

data.SepalWidthCm.apply(lambda n : n/2)
# Defining column using other columns

data["total_width"] = data.SepalWidthCm + data.PetalWidthCm

data.head()
#HIERARCHICAL INDEXING



# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/Iris.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Id","Species"]) 

data1.head(25)
