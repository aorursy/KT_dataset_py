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
happy = pd.read_csv('../input/world-happiness-report-2019.csv')
happy.info()
happy.corr
#correlation map

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(happy.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#plt.show()
happy.head()
happy.tail()
happy.columns
happy.columns = [each.split()[0] + "_" + each.split()[1] if (len(each.split())>1) else each for each in happy.columns]

happy.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

happy.Freedom.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

happy.Generosity.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

happy.plot(kind='scatter', x='Ladder', y='Log_of',alpha = 0.5,color = 'red')

#plt.scatter(happy.Ladder, happy.Log_of, color = 'red')

plt.xlabel('Ladder')              # label = name of label

plt.ylabel('Log_of_GDP\nper_capita')

plt.title('Ladder & Log_of_GDP\nper_capita Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

happy.Corruption.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

happy.Corruption.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['spain'] = "barcelona"    # update existing entry

print(dictionary)

dictionary['france'] = "paris"       # Add new entry

print(dictionary)

del dictionary['spain']              # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary, 'paris' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
# In order to run all code you need to take comment this line

del dictionary         # delete entire dictionary     

#print(dictionary)       # it gives error because dictionary is deleted
happy = pd.read_csv('../input/world-happiness-report-2019.csv')
series = happy['Corruption']        # happy['Corruption'] = series

print(type(series))

#seriler vektör şeklinde uzanan tek boyutlu yapılardır

dataFrame = happy[['Corruption']]  # happy[['Corruption']] = data frame

print(type(dataFrame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = happy['Corruption']> 140     # There are only 8 countries who have higher Corruption value than 140

happy[x] # sadece x değeri (> 140) True olanlarını yazdırır
# 2 - Filtering pandas with logical_and

# There are only 5 countries who have higher Corruption value than 14o and higher Freedom value than 100

happy[np.logical_and(happy['Corruption']>140, happy['Freedom']>100)]

#happy[(happy['Corruption'] > 140) & (happy['Freedom'] > 100)]
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
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

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in happy[['Corruption']][0:1].iterrows():

    print(index," : ",value)





def tuble_ex():

    """ return defined t tuble"""

    t = (1,2,3) #this lines PACKS values into variable t

    return t

a,b,c = tuble_ex() # this lines UNPACKS values of variable t

print(a,b,c)
# Scope 

# guess print what

x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope



# What if there is no local scope

x = 5

def f():

    y = 2*x        # there is no local scope x

    return y

print(f())         # it uses global scope x

# First local scopesearched, then global scope searched, 

# if two of them cannot be found lastly built in scope searched.
#nested function

def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2

print(square())
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print("with def arg: ", f(5))

# what if we want to change default arguments

print("ovverriding def arg: ",f(5,4,3))



# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

print("")

f("one argument with *args: ", 1)

print("")

f("multiple arguments with *args: ", 1,2,3,4)



# flexible arguments **kwargs that is dictionary



def f(**kwargs):

    """ print key and value of dictionary"""

    for i, j in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(i, " ", j)

print("dictionary arguments with **kwargs: ")

f(country = 'spain', capital = 'madrid', population = 123456)
# lambda function / Faster way of writing function

square = lambda x: x**2     # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
#ANONYMOUS FUNCTION

#Like lambda function but it can take more than one arguments.

# map(func,seq) : applies a function to all the items in a list



number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
# iteration example

name = "ronaldo"

it = iter(name)

print(next(it))    # print next iteration

print(*it)         # print remaining iteration
# zip example

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)



#unzip

un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuple

print(un_list1)

print(un_list2)

print(type(un_list2))

print(type(list(un_list1))) # converting un_list1 type from tuple to list 
# Example of list comprehension

num1 = [1,2,3]

num2 = [i + 1 for i in num1 ] # list comprehension

print(num2)



"""

[i + 1 for i in num1 ]: list of comprehension 

i +1: list comprehension syntax 

for i in num1: for loop syntax 

i: iterator 

num1: iterable object

"""

# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
# lets return world-happiness-report-2019.csv and make one more list comprehension 

# example lets classify countries whether they have less or much freedom. 

# Our threshold is average freedom.

threshold = sum(happy["SD of Ladder"])/len(happy["SD of Ladder"])

print("threshold ", threshold)

happy["SD_level"] = ["much" if i > threshold else "less" for i in happy["SD of Ladder"]]

happy.loc[50:60,["SD_level","SD of Ladder"]] # we will learn loc more detailed later
# For example lets look frequency of Social Support

print(happy['Social support'].value_counts(dropna =False))  # if there are nan values that also be counted

# this is not a suitable data frame, please consider more carefully while choosing your exercise data set
happy.describe() #ignore null entries
happy.boxplot(column='SD of Ladder',by = 'Positive affect')

plt.show()
# TIDY DATA

happy_new = happy.head()

happy_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=happy_new,id_vars = 'Country (region)', value_vars= ['Positive affect','Corruption'])

melted
# PIVOTING DATA

# Reverse of melting.

# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Country (region)', columns = 'variable',values='value')
# CONCATENATING DATA

# Firstly lets create 2 data frame

data1 = happy.head()

data2= happy.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
happy1 = happy['SD of Ladder'].head()

happy2= happy['Social support'].head()

conc_happy_col = pd.concat([happy1,happy2],axis =1) # axis = 0 : adds dataframes in row

conc_happy_col
happy.dtypes
# lets convert object(str) to categorical and int to float.

happy['Country (region)'] = happy['Country (region)'].astype('category')

happy['SD of Ladder'] = happy['SD of Ladder'].astype('float')



happy.dtypes
happy.info()
# Lets check Type 2

happy["Corruption"].value_counts(dropna = False) # show the number of NaN values

# As you can see, there are 8 NAN value
# Lets drop nan values

happy1 = happy   # also we will use data to fill missing value so I assign it to data1 variable

happy1["Corruption"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. 

# Changes automatically assigned to data

# So does it work ?

# Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true



# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false



assert happy['Corruption'].notnull().all() # returns nothing because we drop nan values
happy["Corruption"].fillna('empty',inplace = True)



assert happy['Corruption'].notnull().all() # returns nothing because we do not have nan values



# With assert statement we can check a lot of thing. For example

assert happy.columns[0] == 'Country (region)'

assert happy.Ladder.dtypes == np.int
# BUILDING DATA FRAMES FROM SCRATCH



# data frames from dictionary

country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

zipped

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"] = ["madrid","paris"] # Broadcasting: Create new column and assign different value to entire column

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column with same value

df
# Plotting all data 

happy1 = happy.loc[:,["Ladder","Freedom","Corruption"]]

happy1.plot()

# it is confusing
# subplots

happy1.plot(subplots = True)

plt.show()
# scatter plot  

happy1.plot(kind = "scatter", x = "Freedom", y = "Corruption")

plt.show()
# hist plot

happy1.plot(kind = "hist",y = "Freedom",bins = 50,range= (0,160),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

happy1.plot(kind = "hist",y = "Freedom",bins = 50,range= (0,160),normed = True,ax = axes[0])

happy1.plot(kind = "hist",y = "Freedom",bins = 50,range= (0,160),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
happy.describe()
# Indexing Pandas Time Series

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of happiness data and add it a time list

happy2 = happy.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

happy2["date"] = datetime_object

# lets make date as index

happy2= happy2.set_index("date")

happy2
# Now we can select according to our date index

print(happy2.loc["1993-03-16"])

print(happy2.loc["1992-03-10":"1993-03-16"])
# We will use happy2 that we create at previous part

happy2.resample("A").mean()
# Lets resample with month

happy2.resample("M").mean()

# As you can see there are a lot of nan because happy2 does not include all months
# In real life (data is real. Not created from us like happy2) we can solve this problem with interpolate

# We can interpolete from first value

happy2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

happy2.resample("M").mean().interpolate("linear")
# read data

happy = pd.read_csv('../input/world-happiness-report-2019.csv')

#happy.head()

happy = happy.set_index("Ladder")

happy.head()
# indexing using square brackets

happy["Freedom"][1]
# using column attribute and row label

happy.Freedom[1]
# using loc accessor

happy.loc[1,["Freedom"]]
# read data

happy = pd.read_csv('../input/world-happiness-report-2019.csv')

happy['#'] = range(1, len(happy) + 1)

happy = happy.set_index("#")

happy.head()
# indexing using square brackets

happy['Generosity'][2]
# using column attribute and row label

happy.Generosity[2]
# using loc accessor

happy.loc[1,["Generosity"]]
# Selecting only some columns

happy[["Country (region)", "Generosity", "Negative affect"]]
# Difference between selecting columns: series and dataframes

print(type(happy["Generosity"]))     # series

print(type(happy[["Generosity"]]))   # data frames
# Slicing and indexing series

happy.loc[1:10,"Country (region)":"Positive affect"]   # 10 and "Positive affect" are inclusive
# Reverse slicing 

happy.loc[10:1:-1,"Country (region)":"Positive affect"]
# From something to end

happy.loc[1:10,"Freedom":] 
# Creating boolean series

boolean = happy.Corruption > 140

happy[boolean]
# Combining filters

first_filter = happy.Corruption > 140

second_filter = happy.Generosity > 80

happy[first_filter & second_filter]
# Filtering column based others

happy.Corruption[happy.Generosity>150]
# Plain python functions

# all the values at the Corruption column divided by 2

def div(n):

    return n/2

happy.Corruption.apply(div)
# Or we can use lambda function

happy.Corruption.apply(lambda n : n/2)
# Defining column using other columns

happy["average_affect"] = happy["Positive affect"] - happy["Negative affect"]

happy.head()
# our index name is this:

print(happy.index.name)

# lets change it

happy.index.name = "index_name"

happy.head()
# Overwrite index

# if we want to modify index we need to change all of them.

happy.head()

# first copy of our data to data3 then change index 

happy3 = happy.copy()

# lets make index start from 100. It is not remarkable change but it is just example

happy3.index = range(10,166,1)

happy3.head()



# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("#")

# also you can use 

# data.index = data["#"]
# lets read data frame one more time to start from beginning

happy = pd.read_csv('../input/world-happiness-report-2019.csv')

happy.head()

# As you can see there is index. However we want to set one or 

# more column to be index
# Setting index : Country (region) is outer Negative affect is inner index

happy1 = happy.set_index(["Country (region)","Negative affect"]) 

happy1.head(10)

# happy1.loc["Freedom","Corruption"] # how to use indexes
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
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max()
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min()
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()