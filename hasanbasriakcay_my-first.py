# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# print(os.listdir("../input"))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='iso-8859–1')
data.info()
correlation = data.corr()
correlation
correlation.shape
threshold_1 = 0.8

for i in range(0,77):

    temp_col = correlation.columns[i]

    for j in range(0,77):

        if correlation[temp_col][j] > threshold_1 and i != j:

            print(str(temp_col)," - ", str(correlation.columns[j]), " = ", str(correlation[temp_col][j]))
threshold_2 = -0.5

for i in range(0,77):

    temp_col = correlation.columns[i]

    for j in range(0,77):

        if correlation[temp_col][j] < threshold_2 and i != j:

            print(str(temp_col)," - ", str(correlation.columns[j]), " = ", str(correlation[temp_col][j]))
#correlation map

f,ax = plt.subplots(figsize=(100, 100))

sns.heatmap(correlation, annot=True, linewidths=.5, fmt= '.5f',ax=ax)

plt.show()
data.head()
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.nkill.plot(kind = 'line', color = 'g',label = 'nkill',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.country.plot(color = 'r',label = 'country',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = country, y = region

data.plot(kind='scatter', x='country', y='region',alpha = 0.5,color = 'red')

plt.xlabel('country')              # label = name of label

plt.ylabel('region')

plt.title('Country Region Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.nkill.plot(kind = 'hist',bins = 500,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'Turkey' : 'Istanbul','Usa' : 'New York'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['Turkey'] = "Antalya"    # update existing entry

print(dictionary)

dictionary['France'] = "Paris"       # Add new entry

print(dictionary)

del dictionary['Usa']              # remove entry with key 'spain'

print(dictionary)

print('France' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

# print(dictionary)       # it gives error because dictionary is deleted
series = data['nkill']        # data['Defense'] = series

print(type(series))

data_frame = data[['nkill']]  # data[['Defense']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['nkill']>1000     # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['nkill']>1000, data['country']>200 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['nkill']>1000) & (data['country']>200)]
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'Turkey':'Istanbul','France':'Paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['nkill']][0:2].iterrows():

    print(index," : ",value)
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
# zip example

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
# Example of list comprehension

num1 = [1,2,3]

num2 = [(i + 1) for i in num1]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
threshold = sum((i if i >=0 else 0) for i in data.nkill)/len(data.nkill)

print(threshold)

print(len(data.nkill))

data["nkill_level"] = ["Above threshold" if i > threshold else "Under threshold" for i in data.nkill]

data.loc[:80,["nkill_level","nkill"]] # we will learn loc more detailed later
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='iso-8859–1')

data.head()  # head shows first 5 rows
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
# For example lets look frequency of pokemom types

print(data['iyear'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
# For example max HP is 255 or min defense is 5

data.describe() #ignore null entries
data.boxplot(column='nkill',by = 'suicide')
# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new
melted = pd.melt(frame=data_new,id_vars = 'eventid', value_vars= ['country','region'])

melted
melted.pivot(index = 'eventid', columns = 'variable', values='value')
# Firstly lets create 2 data frame

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['country'].head()

data2= data['region'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
# lets convert object(str) to categorical and int to float.

data['country_txt'] = data['country_txt'].astype('category')

data['iday'] = data['iday'].astype('float')
data.dtypes
data.info()
data["country_txt"].value_counts(dropna =False)
# Lets drop nan values

data1=data.copy()   # also we will use data to fill missing value so I assign it to data1 variable

data1["nkill"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
assert 1==1 # return nothing because it is true
assert  data1['nkill'].notnull().all() # returns nothing because we drop nan values
data["nkill"].fillna('empty',inplace = True)
assert  data['nkill'].notnull().all() # returns nothing because we do not have nan values
# data frames from dictionary

country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"] = ["madrid","paris"]

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column

df
# Plotting all data 

data1 = data.loc[:,["country","iyear","nkill"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="country",y = "iyear")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "country",bins = 50,range= (0,50),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "country",bins = 50,range= (0,50),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "country",bins = 50,range= (0,50),normed = True,ax = axes[1],cumulative = True)

# plt.savefig('graph.png')

plt
data.describe()
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

date1 = str(data2.iyear[0]) + "-" + str(data2.imonth[0]) + "-" + str(int(data2.iday[0]))

date2 = str(data2.iyear[1]) + "-" + "1" + "-" + "1"

date3 = str(data2.iyear[2]) + "-" + str(data2.imonth[2]) + "-" + "1"

date4 = str(data2.iyear[3]) + "-" + str(data2.imonth[3]) + "-" + "1"

date5 = str(data2.iyear[4]) + "-" + str(data2.imonth[4]) + "-" + "1"



date_list = [date1,date2,date3,date4,date5]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
# Now we can select according to our date index

print(data2.loc["1970-01-01"])

print(data2.loc["1970-01-01":"1970-07-02"])
# We will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

# it does not change the mean

data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='iso-8859–1')
data.head()
# indexing using square brackets

data["iyear"][0]
data.iyear[1]
# using loc accessor

data.loc[1,["iyear"]]
# Selecting only some columns

data[["iyear","imonth"]]
# Difference between selecting columns: series and dataframes

print(type(data["iyear"]))     # series

print(type(data[["iyear"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"iyear":"iday"]   # 10 and "Defense" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"iyear":"iday"] 
# From something to end

data.loc[1:10,"iyear":] 
# Creating boolean series

boolean = data.iyear > 2000

data[boolean]
# Combining filters

first_filter = data.iyear > 2000

second_filter = data.iyear < 2005

data[first_filter & second_filter]
# Filtering column based others

data.country_txt[data.iyear>2016]
# Plain python functions

def div(n):

    return n/2

data.nkill.apply(div)
# Or we can use lambda function

data.nkill.apply(lambda n : n/2)
# Defining column using other columns

#data["nkill"].fillna(0.0,inplace = True)

#data["nkillus"].fillna(0.0,inplace = True)

#print(type(data.iyear[0]))

#print(type(data.iday[0]))

data["a_value"] = data.iyear + data.iyear

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,181791,1)

data3.head()
data.head()
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["iyear","imonth"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)
# level determines indexes

df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max() 
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 