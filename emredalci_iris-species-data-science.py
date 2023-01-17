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
data = pd.read_csv("../input/iris/Iris.csv") 
data.info() 
data.corr()
#correlation map

f,ax = plt.subplots(figsize =(18,18))

sns.heatmap(data.corr(), annot= True,linewidths = 0.5, fmt = "0.2f",ax = ax)

# annot = If True, write the data value in each cell. fmt =String formatting code to use when adding annotations.

plt.show()
data.head(10) # return first 10 rows
data.columns
# Line Plot

#color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.SepalLengthCm.plot(kind = "line", color ="green", label = "SepalLengthCm", linewidth = 1, alpha = 0.7, grid = True, linestyle = ":")

data.PetalLengthCm.plot(kind = "line", color = "red", label = "PetalLengthCm", linewidth =1 , alpha = 0.7, grid = True, linestyle = "-.")

plt.legend() # legend = puts label into plot

plt.xlabel("x axis") # label = name of label

plt.ylabel("y label") # label = name of label

plt.title("Line Plot") #title = title of plot

plt.show()
# Scatter Plot 

# x = PetalLengthCm, y = SepalLengthCm

data.plot(kind ="scatter", x = "PetalLengthCm",y ="SepalLengthCm", alpha = 0.7, color= "red")

plt.title("SepalLengthCm-PetalLengthCm Scatter Plot") #title = title of plot

# Histogram

# bins = number of bar in figure

data.PetalLengthCm.plot(kind = "hist", bins = 50, figsize = (12,12))

plt.title("Histogram of PetalLengthCm")
# clf() = cleans it up again you can start a fresh

data.PetalLengthCm.plot(kind = "hist", bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {"Poland":"Poznan","Germany":"Berlin","Turkey":"Istanbul"}

print(dictionary.keys())

print(dictionary.values())

print(type(dictionary))
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary["Poland"] = "Warsaw" #update existing entry

print(dictionary)

dictionary["France"] = "Paris" # Add new key and value

print(dictionary)

del dictionary["Germany"] # remove entry with key "Germany"

print(dictionary)

print("Turkey" in dictionary)  # check include or not

dictionary.clear()  # remove all entries in dict

print(dictionary)
data = pd.read_csv("../input/iris/Iris.csv")
series = data["SepalLengthCm"]   # data['SepalLengthCm'] = series

print(type(series))

data_frame = data[["SepalLengthCm"]]  # # data[['SepalLengthCm']] = data frame

print(type(data_frame))
# Filtering Pandas data frame

x = data["SepalLengthCm"] > 7 # There are 12 flowers who have higher SepalLengthCm value than 7

data[x]
# Filtering pandas with logical_and

data[np.logical_and(data["SepalLengthCm"]>7, data["PetalLengthCm"] > 6 )]

# There are 9 flowers who have higher SepalLengthCm value than 7 and higher PetalLengthCm value than 6
# This is also same with previous code line. Therefore we can also use '&' for filtering

data[(data["SepalLengthCm"] > 7) & (data["PetalLengthCm"] > 6)]
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i!= 5:

    print("i is ",i)

    i +=1

print(i,"is equal to 5")
# Stay in loop if condition( i is not equal 5) is true

liste = [-1,-2,-3,-4,-5]

for i in liste :

    print("i is ",i)

print("")



# Enumerate index and value of list

# index : value = 0:-1, 1:-2, 2:-3, 3:-4, 4:-5

for index,value in enumerate(liste):

    print(index,":",value)

print("")



#For dictionary we can reach index and value

for key,value in dictionary.items():

    print(key,":",value)

print("")



#For pandas we can reach index and value

for index,value in data[["SepalLengthCm"]][0:1].iterrows():

    print(index,":",value)

print("")
# example of what we learn above

def tuble_ex():

    """"return defined t tuble"""

    t= (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# guess print what

x = 2

def f():

    x = 3

    return x

print(x) # x = 2 global scope

print(f()) # x = 3 local scope
# What if there is no local scope

x = 5

def f():

    y = x*x # there is no local scope x

    return y

print(f()) # it uses global scope x

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope

import builtins

dir(builtins)
# nested function

def square():

    """" return square of func"""

    def add():

        x = 2

        y = 3

        z = x+y

        return z

    return add()**2

print(square())
# default arguments

def f(a, b= 1, c = 2):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key,value in kwargs.items(): # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key,":",value)

f(country = 'spain', capital = 'madrid', population = 123456)
# Lambda Function

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
number_list = [10,20,30]

y = map(lambda x:x**2, number_list)

print(list(y))
# iteration example

name = "ISTANBUL"

it = iter(name)

print(next(it))  # print next iteration

print(*it)       # print remaining iteration
# zip example

list1 = [10,20,30,40]

list2 = [50,60,70,80]

z = zip(list1,list2)

print(z)

newlist = list(z)

print(newlist)
unzip = zip(*newlist)

unlist1,unlist2 = list(unzip)  # unzip returns tuble

print(unlist1)

print(unlist2)

print(type(unlist1))
#[2*i+i**2 for i in num1 ]: list of comprehension

#i +1: list comprehension syntax

#for i in num1: for loop syntax

#i: iterator

#num1: iterable object

num1 = [10,20,30]

num2 = [2*i+i**2 for i in num1 ]

print(num2)
# Conditionals on iterable

num3 = [10,20,30]

num4 = [i**2 if i==10 else i-5 if i == 20 else i+5 for i in num3]

print(num4)
# lets classify flowers whether they have high or low Sepal Length. Our threshold is average Length.

threshold = sum(data.SepalLengthCm)/len(data.SepalLengthCm)

data["Sepal_Length"] = ["high" if i>threshold else "low" for i in data.SepalLengthCm]

data.loc[::-10,["Sepal_Length","SepalLengthCm"]]
data = pd.read_csv("../input/iris/Iris.csv")

data.head() # head shows first 5 rows
data.tail() # tail shows last 5 rows
data.columns # columns gives column names of features
data.shape
data.info() # info gives data type, number of sample or row, number of feature or column, feature types and memory usage
print(data["Species"].value_counts(dropna = False)) #For example lets look frequency of Species

# if there are nan values that also be counted

data.describe() #ignore null entries
data.boxplot(column = "SepalLengthCm", by = "Species")

#compare SepalLengthCm of flowers that are Iris-virginica, Iris-setosa or Iris-versicolor
# Firstly I create new data from Iris Species data to explain melt nore easily.

data_new = data.head() # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame = data_new , id_vars = "Species", value_vars = ["PetalWidthCm", "PetalLengthCm"])

melted
df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',

                            'two'],

                    'bar': ['A', 'B', 'C', 'A', 'B', 'C'],

                    'baz': [1, 2, 3, 4, 5, 6],

                    'zoo': ['x', 'y', 'z', 'q', 'w', 't']})

df
df.pivot(index='foo', columns='bar', values='baz')

#Return reshaped DataFrame organized by given index / column values.
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis = 0,ignore_index = True) # axis = 0 : adds dataframes in row

conc_data_row
data3 = data["SepalLengthCm"].head()

data4 = data["PetalLengthCm"].head()

conc_data_col = pd.concat([data3,data4],axis =1,ignore_index = False) # axis = 0 : adds dataframes in col

conc_data_col
data.dtypes
# lets convert object(str) to categorical and int to float.

data["Species"] = data["Species"].astype("category") #object(str) to categorical

data["Id"] = data["Id"].astype("float") #int to float.

data.dtypes
data.info()
data["SepalLengthCm"].value_counts(dropna =False)
# If we would have some NaN values, we could drop these values. For example :

df1 = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],

                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],

                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),

                             pd.NaT]})

df1
df1.dropna(inplace=True) # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

df1
assert df1["toy"].notnull().all()  # returns nothing because we drop nan values
# data frames from dictionary

country = [ "Germany","Turkey"]

population = ["10","11"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
df["capital"] = ["berlin","ankara"]

df
df["income"] = 0 #Broadcasting entire column

df
# plot data 

data1 = data.loc [:,["SepalLengthCm","PetalLengthCm"]]

data1.plot()
# subplot

data1.plot(subplots = True)

plt.show()
# scatter plot

data1.plot(kind ="scatter", x = "SepalLengthCm", y = "PetalLengthCm")

plt.show()
# hist plot

data1.plot ( kind = "hist",y = "SepalLengthCm", bins = 50, normed = True)

plt.show ()
# histogram subplot with non cumulative and cumulative

#bins: number of bins , range(tuble): min and max values of bins, normed(boolean): normalize or not , cumulative(boolean): compute cumulative distribution

fig , axes = plt.subplots (nrows = 2, ncols = 1)

data1.plot(kind = "hist", y = "SepalLengthCm", bins = 50 , normed = True, ax = axes[0] ) 

data1.plot(kind ="hist", y ="SepalLengthCm", bins = 50, normed = True, ax = axes[1], cumulative = True)

plt.savefig("graph.png")

plt
time_list = ["1995-10-05","1995-12-06"]

print(type(time_list[1])) #date is string

datetime_object = pd.to_datetime(time_list) # from list to datetime

print(type(datetime_object)) 
# In order to practice lets take head of Iris data and add it a time list

data2= data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime1_object = pd.to_datetime(date_list)

data2["date"] = datetime1_object

data2 = data2.set_index("date") # make date as index

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part

data2.resample("A").mean()
data2.resample("M").mean() #there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

data2.resample("M").mean().interpolate("linear")
# read data

data = pd.read_csv("../input/iris/Iris.csv")

data = data.set_index("Id")

data.head()
# indexing using square brackets

data["SepalLengthCm"][1]
# using column attribute and row label

data.SepalLengthCm[1]
# using loc accessor

data.loc[1,["SepalLengthCm"]]
# Selecting only some columns

data[["SepalLengthCm","SepalWidthCm"]]
# Difference between selecting columns: series and dataframes

print(type(data["SepalLengthCm"]))

print(type(data[["SepalLengthCm"]]))
# Slicing and indexing series

data.loc[1:10,"SepalLengthCm":"SepalWidthCm"]
# Reverse slicing

data.loc[10:1:-1,"SepalLengthCm":"SepalWidthCm"]
# From something to end

data.loc[1:10,"SepalLengthCm":]
boolean = data.SepalWidthCm >3.5

data[boolean]
# Combining filters

first_filter = data.SepalWidthCm > 3.5

second_filter = data.PetalWidthCm > 0.3

data[first_filter & second_filter]
# Filtering column based others

data.SepalLengthCm[data.PetalLengthCm > 6]
# Plain python functions

def div(n):

    return n/2

data.SepalLengthCm.apply(div)
# Or we can use lambda function

data.SepalLengthCm.apply(lambda n: n/2)
# Defining column using other columns

data["total_length"] = data.SepalLengthCm + data.PetalLengthCm

data.head()
# our index name is this:

print(data.index.name)
data.index.name = "index_name"

data.head()
!!!!!!!!!# Overwrite index

data3 = data.copy()

data3.index = range(0,150,5)

data3.head()
data = pd.read_csv("../input/iris/Iris.csv")

data.head() # # As you can see there is index. However we want to set one or more column to be index
data1 = data.set_index(["SepalLengthCm","PetalLengthCm"])

data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# according to treatment take means of other features

df.groupby("treatment").mean() # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()