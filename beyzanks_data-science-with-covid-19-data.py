# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

from datetime import datetime #datetime module



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head(10) # first 10 rows
df.info() # information about datas
df.tail() # default -> last 5 rows
df.columns # names of data columns
df.count() # data count in columns
df['Province/State'].value_counts(sort=True,ascending=True) # sort Province/State value in ascending order 
df.size # returns size of dataframe which is equivalent to total number of elements. That is rows x columns.
f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.1f',cbar=True,ax=ax)

plt.show()
df.Deaths.plot(kind = 'line', color = 'red',label ='Deaths',linewidth=1,alpha = 0.5,grid = True,linestyle = '--')

plt.legend(loc='upper right') # puts label into plot

plt.xlabel('x axis')    # name of xlabel

plt.ylabel('y axis')    # name of xlabel

plt.title('Line Plot')  # title of plot

plt.show()
df.plot(kind='scatter', x="Deaths", y='Recovered', alpha=0.4, color='blue')

plt.xlabel('Deaths') # name of xlabel

plt.ylabel('Recovered') # name of ylabel

plt.title('Deaths Recovered Scatter Plot') #title of plot

plt.show()



df.Deaths.plot(kind = 'hist',bins = 100,figsize = (15,15))

plt.title("Histogram")

plt.show()
df.plot(x='ObservationDate',y='Recovered',color = 'green',label ='Recovered',linewidth=1,alpha = 0.5,grid = True,linestyle = '--')

plt.title('', color='black')

plt.xticks(rotation = 90) # rotates the labels 90 degrees.

#plt.tight_layout() 

plt._show()
#Other date plot

'''df['ObservationDate'] = df['ObservationDate'].map(lambda x: datetime.strptime(str(x), '%m/%d/%Y'))

x = df['ObservationDate']

y = df['Recovered']



plt.plot(x,y,color='pink')# plot

plt.gcf().autofmt_xdate()# beautify the x-labels

plt.show()'''
currency = {

    "Dolar" : "USD",

    "Türk Lirası" : "TR",

    "Euro" : "EUR",

    "Sterlin" : "GBP"

}

print('My dictionary :',currency)

print(currency.keys())

print(currency.values())
x = currency.get('Euro') # get 'Euro''s value

print(x)
for k,v in currency.items(): # print key and value in dictionary

    print(k+' : '+v)
if "Sterlin" in currency: # check 'Sterlin' in dictionary

  print("Yes, 'Sterlin' is one of the keys in the currency dictionary")
currency['Kanada Doları']='CAD' # adding item

print(currency)
#currency.clear() #remove dictionary

#del currency # delete dictionary 
df[:8] # 0-8 rows
#Filtering

x = df['Deaths']>5000 

df[x]
df[(df['Recovered']>5000) & (df['Deaths']<500)]
(df.groupby(['ObservationDate','Country/Region']).sum().loc[lambda df: df['Deaths'] > 4000]) # data selection (date format -> %m %d %Y)
df.sample(n=6, weights='Deaths') # selecting random samples
i=0



while True:

    print(i,"Data Science")

    i +=1

    if i==6:

        break    #break the loop
#print columns names with while loop

i=0

while i<len(df.columns):

    print("Column",i, ':' ,df.columns[i])

    i +=1

#print columns names with for loop

for col in df.columns:

    print(col)
#For pandas we can achieve index and value

for index,value in df[['Province/State']][0:5].iterrows():

    print(index," : ",value)
# iterate over rows with iterrows()

for index, row in df.head(6).iterrows():

     # access data using column names

     print(index+1, row['Province/State'], row['Country/Region'])
#Create Tuple

x = ("C","Java","Python")



print(x)
x = ("C","Java","Python")

y = list(x)

y[0] = "C++"

x = tuple(y)



print(x)
#Values of tuple with for loop

for i in x:

    print(i)



print("\n")

(l1,l2,l3)=x

print("Values:",l1,l2,l3)
x = 2 #Global scope



def f(y):

    result = y**x 

    return result



print(f(5))
count = 1



def func():

    for count in range(6):

        count +=1

    return count



print(count) # count = 1 global scope

print(func()) # count = 6 local scope

# How can we learn what is built in scope

import builtins

dir(builtins)
def function1(): # outer function

    print ("Hello from outer function")

    def function2(): # inner function

        print ("Hello from inner function")

    function2()



function1()
# example finding number's square with nested function

def func1(x):

    def func2():

        result = x**2

        return result

    return func2()



#number = int(input("Please enter a number ")) with input value -> func1(number)

print(f"{5}'s square =",func1(5))
# default argument

def func(lang='Python'):

    return lang



print("Programing language:" ,func())

print("Programing language:", func('C++'))
# flexible arguments *args

def func(*args):

    for i in args:

        print(i)



func(3,5,8,11)

list1=[1,2,3,4,5,6]

func(list1)

list2=[1,2,3,4,5,6],[2,5,7,8,9,10]

func(list2)
# flexible arguments **kwargs that is dictionary

def func(**kwargs):

    for key, value in kwargs.items():

        print(f"{key} -> {value}")

        

func(Class = 'Data Science', Part = '2')

        
# lambda function

negative = lambda x : -x

print(negative(5))



result = lambda a,b,c : a*b*c

print(result(2,5,10))
import math # for sqrt function



num_list=[0,4,16,36]

func = map(lambda x:math.sqrt(x),num_list)

print(tuple(func))
# iteration example

subject = "DataScience"

it=iter(subject)

print(next(it))

print(next(it))

print(next(it))

print(next(it))

print(*it)
# zip example zip() -> zip lists

list1=["Dolar","Türk Lirası","Euro","Sterlin"]

list2=["USD","TR","EUR","GBP"]

z = tuple(zip(list1,list2))

print(z)
un_zip = zip(*z)

un_list1,un_list2 = tuple(un_zip)

print(un_list1)

print(un_list2)
list1 = [1,2,5,8,14,21]

list2 = [print(f"{i}: Odd") if i%2!=0 else print(f"{i}: Even") for i in list1]
# list comprehension for covid_19_data dataset

threshold = sum(df.Deaths)/len(df.Deaths) # average Deaths

print("Average Deaths:",threshold) 

# List values between 60000 and 80000 according to deaths level(high or low)

df["Deaths_Level"] = ["High" if i > threshold else "Low" for i in df.Deaths] 

df.loc[60000:80000,["Deaths_Level","Deaths","Country/Region"]] 
df.head(15) #first 15 data
df.tail(15) #last 15 data
df.shape # (row,column)
df.info() # information about datas
# For example lets look frequency of countries

print(df['Country/Region'].value_counts(dropna =False))
df.describe() # There may be problems with statistics account because there are too many 0 values.
df.mask(df == 0).describe() # 0 values are masked.
# For example: compare deaths covid-19 that are deaths_level high or not

# Black line at top is max

# Blue line at top is 75%

# Green line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

df.boxplot(figsize=(15,15),column='Deaths',by = 'Deaths_Level') # There may be problems because there are too many 0 values and data.
# Firstly I create new data from covid_19 data to explain melt more easily.

data_new = df.loc[55591:55601]    # I only take 10 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Country/Region', value_vars= ['Deaths','Recovered'])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot_table(index = 'Country/Region', columns = 'variable',values='value')
data1 = df.head()

data2= df.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row # Concatenating data1 and data2 
data1 = df['Country/Region'].tail()

data2= df['Deaths'].tail()

data3= df['Recovered'].tail()

conc_data_col = pd.concat([data1,data2,data3],axis =1) # axis = 1 : adds dataframes in column

conc_data_col
df.dtypes
# lets convert object(str) to categorical and float to int.

df['Deaths_Level'] = df['Deaths_Level'].astype('category')

df['Confirmed'] = df['Confirmed'].astype('int')



df.dtypes
# Lets look at does covid_19 data have nan value

# As you can see there are 68558 entries. However Province/State has 44125 non-null object so it has 24433 null object.

df.info()
# Lets chech Province/State

df["Province/State"].value_counts(dropna =False)

# As you can see, there are 24433 NAN value
# Lets drop nan values

data1=df   # also we will use df to fill missing value so I assign it to data1 variable

data1["Province/State"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
# Lets check with assert statement

# Assert statement:

assert  df["Province/State"].notnull().all() # returns nothing because we drop nan values
df["Province/State"].fillna('empty',inplace = True)
assert  df["Province/State"].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

assert df.columns[0] == 'SNo' #True

#assert df.Deaths_Level.dtypes == np.int #False
df["Province/State"].value_counts(dropna = False) # now there isn't nan value in table
# data frames from dictionary

language = ["English","German","Turkish"]

level = ["B2","B1","C2"]

list_label = ["language","level"]

list_col = [language,level]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

print(data_dict)

data = pd.DataFrame(data_dict)

data
# Add new columns with list comprehension 

data["completed"]=["Yes" if i == "C2"  else "No" for i in data.level]

data
# Broadcasting

data["necessary"] = "Yes" #Broadcasting entire column

data
# Plotting all data 

data1 = df.loc[:,["Confirmed","Deaths","Recovered"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Confirmed",y = "Deaths",alpha=0.4,color="red")

plt.show()
# histogram plot  

data1.plot(kind = "hist",y = "Deaths",bins = 30,range= (0,500),density = True) # density was used instead of normed
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Deaths",bins = 50,range= (0,500),density = True,ax = axes[0],color="yellow")

data1.plot(kind = "hist",y = "Deaths",bins = 50,range= (0,500),density = True,ax = axes[1],color="yellow",cumulative = True) #cumulative total

plt.savefig('graph.png')

plt
df.describe()
time_list = ["2020-11-29","2020-07-30"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")



# In order to practice lets take covid-19 data and add it a time list

data2 = df

datetime_object = pd.to_datetime(data2.ObservationDate) # convert ObservationDate that is object to pandas time series

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2.head(10)
# Now we can select according to our date index

print(data2.loc["2020-06-11"])

print(data2.loc["2020-06-11":"2020-07-30"])
data2.resample("M").mean() # average covid-19 results by months
#data2.resample("M").first().interpolate("linear")

#data2.resample("M").mean().interpolate("linear")

# we didn't use interpolate because data already include all months
# read data

data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data= data.set_index("SNo")

data.head()
# indexing using square brackets

data["Country/Region"][55543]
# using column attribute and row label

data.Deaths[55543]
# using loc accessor

data.loc[1,["Country/Region"]]
# selecting only some columns

data[["Country/Region","Deaths","Recovered"]]
# difference between selecting columns: series and dataframes

print(type(data["Confirmed"]))     # series

print(type(data[["Confirmed"]]))   # data frames
# slicing and indexing series

data.loc[55000:55010,"Country/Region":"Deaths"]   # 10 and "Deaths" are inclusive
# reverse slicing 

data.loc[55010:55000:-1,"Country/Region":"Deaths"]
# from something to end

data.loc[55000:55010,"Deaths":] 
# creating boolean series

boolean = data.Deaths > 40000

data[boolean]
# combining filters

first_filter = data.Deaths < 5000

second_filter = data.Recovered > 50000

data[first_filter & second_filter]
# filtering column based others

data.Deaths[data.Recovered>20000]
# plain python functions

def square(n):

    return n**2

data.Confirmed.apply(square)
# or we can use lambda function

data.Confirmed.apply(lambda n : n**2)
# defining column using other columns

data["gap"] = data.Recovered - data.Deaths

data.loc[55000:55010]
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "Index"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,233710,2)

data3.head()
# data= data.set_index("#") or  data.index = data["#"]
# lets read data frame one more time to start from beginning

data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : Country/Region is outer Province/State is inner index

data1 = data.set_index(["Country/Region","Province/State"]) 

data1.head(10000)
dic = {"job":["Engineer","Engineer","Chef","Chef"],"gender":["M","F","M","F",],"experience":[0,5,12,8],"age":[22,28,40,32]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="job",columns = "gender",values="age")
df1 = df.set_index(["job","gender"])

df1
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="job",value_vars=["age","experience"])
df
# according to job take means of other features

df.groupby("job").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("job").age.max() 
# Or we can choose multiple features

df.groupby("job")[["age","experience"]].min() 
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["job"] = df["job"].astype("category")

#df.info()