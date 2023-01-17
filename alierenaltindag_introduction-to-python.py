import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

data = pd.read_csv("../input/Pokemon.csv") # Import dataset

data.head(5)
data.HP.plot(kind='line',grid=True,label='HP',color="red",linestyle='-',alpha=0.5,linewidth=1) # Hp

data.Attack.plot(kind='line',color='green',grid=True,label='Attack',linestyle='-',alpha=0.5,linewidth=1) # Attack

plt.legend(loc='upper left') # We use this function for distinguish lines.

plt.xlabel('HP')

plt.ylabel('Attack')

plt.show()
data.plot(kind='scatter',x = 'HP',y = 'Attack',alpha = 0.7,color = 'red')

plt.xlabel("HP")

plt.ylabel('Attack')

plt.title("Hp Attack Scatter Plot")

plt.show()
data.Defense.plot(kind='hist',bins=100,figsize=(8,5))

plt.show()
dictionary = {"Key1":"Value1", "Key2":"Value2", "Key3":"Value3"}

print(dictionary.keys())

print(dictionary.values())
print(dictionary["Key1"])



dictionary["Key2"] = "none"

print(dictionary["Key2"])



dictionary["Key4"] = "Value4"

print(dictionary["Key4"])



dictionary.clear()

print(dictionary)
series = data["Attack"] # data["Attack"] = Series

data_frame = data[["Attack"]] # data[["Attack"]] = Data_Frame

print(type(series),"\n",type(data_frame))
data.head(5)
filtering = data["Attack"]>170

data[filtering]
data[(data["Attack"]>150) & (data["HP"]>100)]

data[np.logical_and(data["Attack"]>150, data["HP"]>100)] # Both are same thing
list = [1,2,3,4,5]

for i in list:

    print(i)
for index,value in enumerate(list): # We use enumerate method for print both index and value

    print(index,":",value)
dictionary = {"Key1":"Value1","Key2":"Value2","Key3":"Value3"}

for key,value in dictionary.items():

    print(key,":",value)
for index,value in data[["Attack"]][0:3].iterrows():

    print(index, " : ",value)
def udf():

    t = (1,2,3) # We defined a tuple named t

    return t # We return 3 values through tuple

a,b,c = udf()

print(a,b,c)
x = 3 # Global variable

def scope():

    x = 1 # Local variable

    return x

print(x)

print(scope())

print("")

# Global variables can access by everywhere but Local variables can only accessible from local



# If there is no local variable you will access to global variable. for example:

y = 3

def scope2():

    z = 3*y # y variable is global variable

    return z

print(scope2())



# Extra Information: You can't use built in scope names and Python's preset method names as variable names.

# To see built in scopes:

import builtins

dir(builtins) 
def sqrt():

    def total():

        a = 10

        b = 6

        c = a+b

        return c

    return total()**0.5 # We take the sqrt of the value returned from the total function



print(sqrt())
# Default Arguments:

def f(x,y = 2):

    return x+y

print(f(1)) # I didn't assign a value to variable y

print(f(1,4)) # We can change the default value of variable y
# Flexible Arguments

# We use flexible arguments for send as many values as we want to a function. For example:

def f(*args):

    for i in args:

        print(i)

f(1)

f(1,2,3)

print("")



# Also We use flexible arguments for send as many dictionary items as we want to a function. For example:

def g(**kwargs):

    for key, value in kwargs.items():

        print(key,":",value)

g(key1 = "value1",key2 = "value2",key3 = "value3")
# We use Lambda Function for define a function easily. For example:

f = lambda x : x**2 # First, We defined x variable and this function will return square of x

print(f(3))



g = lambda x,y : x+y # First, We defined x, y variable and this function will return sum of x and y

print(g(1,3))
# map(function, values) # If you want to send multiple values. You must send them in the list

y = list(map(lambda x:x**2,[1,2,3])) # We must convert map function to list format. Because it will return multiple values

print(y)

# If you get an error in this code, don't mind because the error may be caused by kaggle
x = [1,2,3,4,5,6]

it = iter(x)

print(next(it)) # Print next iteration

print(next(it))

print(*it) # Print remaining iteration

print("")



string = "asdfg"

it2 = iter(string)

print(next(it2)) # Print next iteration

print(*it2) # Print remaining iteration



# All objects that we can use with loops are iterable objects
# Zip

list1 = [1,2,3,4]

list2 = [5,6,7,8] # the length of the first list must be equal to the length of the second list

f = list(zip(list1,list2)) # We must convert zip function to list format.



# Unzip

unzip = zip(*f)

unlist1,unlist2 = list(unzip) # Unzip returns tuple

print(unlist1)

print(unlist2)

# If you get an error in this code, don't mind because the error may be caused by kaggle
list1 = [1,2,3,4] # Lists are iterable object

f = [i**2 for i in list1]

print(f)

print("")



# Conditional List Comprehension

list2 = [10,15,20,25]

g = [i if i%10==0 else i**2 if i==15 else i+3 for i in list2]

print(g)

# Both is same

# def g(*args):

#     for i in args:

#         if(i % 10 == 0):

#             print(i)

#         elif(i == 15):

#             print(i**2)

#         else:

#             print(i+3)

# g(10,20,36,64)
avg = sum(data.Attack)/len(data.Attack) # We found the average attack

data["Avg_Attack"] = ["High" if i > avg else "Low" for i in data.Attack] # If pokemon's attack is higher than average, Avg_Attack will be High. If not Avg_Attack will be Low

print(avg)

data.head(5)
# For example lets look frequency of pokemom types

print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
data.describe() # ignore null entries
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='Attack',by = 'Legendary')

plt.show()
# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head(5) # I only take 5 rows into new data

melted = pd.melt(frame = data_new,id_vars = 'Name',value_vars= ['Attack','Defense'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis = 0,ignore_index = True)

conc_data_row
data1 = data['Attack'].head()

data2= data['Defense'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
# lets convert object(str) to categorical and int to float.

data["Type 1"] = data["Type 1"].astype('category')

data['Speed'] = data['Speed'].astype('float')

data.dtypes
data.info()
# Lets check Type 2

data["Type 2"].value_counts(dropna =False)

# As you can see, there are 386 NAN value
# Lets drop nan values

data1=data   # also we will use data to fill missing value so I assign it to data1 variable

data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

# assert data.columns[1] == 'Name'

# assert data.Speed.dtypes == np.int
# data frames from dictionary

country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
df["income"] = 0
data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

plt.show()
data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x="Attack",y="Defense")

plt.show()
data1.plot(kind="hist",y="Defense",bins = 50,range=(0,250),normed = True)

plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.head()
time_list = ["1992-03-08","1992-04-12"]

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
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolate from first value

data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/Pokemon.csv')

data = data.set_index("#")

data.head()
data["HP"][1]
data.loc[1,["HP"]]
data[["HP","Attack"]]
print(type(data["HP"]))

print(type(data[["HP"]]))
data.loc[1:5,["HP","Defense"]]
# From something to end

data.loc[1:10,"Speed":]
data["HP"][data["Speed"]<15]
# Plain python functions

def div(n):

    return n/2

data["HP"].apply(div)
data["HP"].apply(lambda n : n/2)
data["total_power"] = data.Attack + data.Defense

data.head()
print(data.index.name)

# Let's change it

data.index.name = "index_name"

print(data.index.name)
data.head()

data3 = data.copy()

data3.index = range(100,900)

data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("#")

# also you can use 

# data.index = data["#"]
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/Pokemon.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
data1 = data.set_index(["Type 1","Type 2"])

data1.head(20)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1
# levels determines indexes

df1.unstack(level=0)
df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars = "treatment",value_vars = ["age","response"])
df
df.groupby("treatment").mean()
df.groupby("treatment")["age","response"].mean()