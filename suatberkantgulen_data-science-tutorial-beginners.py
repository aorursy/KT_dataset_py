# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files 

#in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.info() # Display the content of data
# To look first 5 values

data.head() 



#data.head(10) # To look first 10 values
# To look last 5 values

data.tail() 



#data.head(15) # To look last 15 values
# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, 

#excluding NaN values.

data.describe()
# Display positive and negative correlation between columns

data.corr()
# Display positive and negative correlation between columns

figure ,axes = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', axes = axes)

plt.show()



# Detailed explanation

# f -> figure to be created

# ax -> a matplotlib.axes.Axes instance to which the heatmap is plotted. If not provided, 

#use current axes or create a new one.

# plt -> matplotlib.pyplot library impoted as plt

# subplots -> type of library feature to be used, can be called to plot two or more plots

#İn one figure.

# figsize -> size of each cells in created table



# figsize - image size

# data.corr() - Display positive and negative correlation between columns

# annot=True -shows correlation rates

# linewidths - determines the thickness of the lines in between

# cmap - determines the color tones we will use

# fmt - determines precision(Number of digits after 0)

# if the correlation between the two columns is close to 1 or 1, the correlation between the two columns has a positive ratio.

# if the correlation between the two columns is close to -1 or -1, the correlation between the two columns has a negative ratio.

# If it is close to 0 or 0 there is no relationship between them.
# To look first 10 values which defense value is the best.

data.sort_values("Defense", ascending = False).head(10)
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1, alpha = 0.5, grid = True, linestyle = ':', figsize=(15,5))

data.Defense.plot(kind = 'line', color = 'r',label = 'Defense',linewidth=1, alpha = 0.5, grid = True, linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# subplots

data.plot(subplots = True, figsize=(15,15))

plt.show()
plt.subplot(4,2,1)

data.HP.plot(kind="line", color="orange", label="HP", linewidth=1, alpha=1, grid=True, figsize=(20,15))

data.Attack.plot(kind="line", color="purple", label="Attack", linewidth=1, alpha=0.5, grid=True)

plt.ylabel("HP")

plt.subplot(4,2,2)

data.Attack.plot(kind="line", color="blue", label="Attack", linewidth=1, alpha=0.8, grid=True, linestyle=":")

plt.ylabel("Attack")

plt.subplot(4,2,3)

data.Defense.plot(kind="line", color="green", label="Defense", linewidth=1, alpha=0.6, grid=True, linestyle="-.")

plt.ylabel("Defense")

plt.subplot(4,2,4)

data.Speed.plot(kind="line", color="red", label="Speed", linewidth=1, alpha=0.4, grid=True)

plt.ylabel("Speed")

plt.show()
# Scatter Plot 

# x = attack, y = defense

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.plot(kind='scatter', x='Attack', y='Defense', alpha = 0.5, color = 'blue', figsize=(10,5))

plt.xlabel('Attack') # label = name of label

plt.ylabel('Defence')

plt.title('Attack - Defense Scatter Plot') # title = title of plot

plt.show() # for showing plot
# Histogram

# bins = number of bar in figure

data.Speed.plot(kind = 'hist',bins = len(data[["Speed"]]), figsize = (12,12))

plt.show()
# To look first 30 values in bar display

data.Attack.head(30).plot(kind="bar", figsize=(10,5))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()

plt.show()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'brand' : 'ford','model' : 'mustang'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['brand'] = "Ford"         # Update existing entry

print(dictionary)



dictionary['year'] = 1964            # Add new entry

print(dictionary)



del dictionary['brand']              # Remove entry with key 'spain'

print(dictionary)



print('model' in dictionary)         # Check include or not



dictionary.clear()                   # Remove all entries in dict

print(dictionary)



# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

#print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/pokemon.csv') # Add csv file as a pandas dataframe
print(type(data))                  # pandas.core.frame.DataFrame

print(type(data[["Attack"]]))      # pandas.core.frame.DataFrame

print(type(data["Attack"]))        # pandas.core.series.Series

print(type(data["Attack"].values)) # numpy.ndarray
series = data['Defense']        # data['Defense'] = series

data_frame = data[['Defense']]  # data[['Defense']] = data frame



print(type(series))

print(type(data_frame), end = "\n\n")



print(series.head(10), end = "\n\n")

print(data_frame.head(10))
# Comparison operator

print(1 >0)

print(1 != 0)



# Boolean operators

print(True and False)

print(True or False)
# Filtering Pandas data frame

Filtered_Defense_200 = data['Defense'] > 200     # There are only 3 pokemons who have higher defense value than 200

data[Filtered_Defense_200]
# Filtering pandas with logical and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['Defense'] > 200, data['Attack'] > 100)]



# This is also same with previous code line. Therefore we can also use '&' for filtering.

#data[(data['Defense'] > 200) & (data['Attack'] > 100)]
# Stay in loop if condition (counter is not equal 10) is true

counter = 0

while counter != 10 :

    print('counter is: ',counter)

    counter +=1 

print('counter is equal to 10 (Loop finished)')
# Stay in loop if condition is true

list_names = ["berkant", "dogus", "kutay"]

for name in list_names:

    print("Name is: ", name)

    

print("")



# Stay in loop if condition is true

list_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for number in list_numbers:

    print("Number is: ", number)



print("")



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(list_numbers):

    print(index,". index : ",value, sep = "")



print("")



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary_car = {'Brand':'Ford','Model':'Mustang'}

for key in dictionary_car:

    print(key)



print("")



for key, value in dictionary_car.items():

    print(key," : ",value)

    

print("")



# For pandas we can achieve index and value

for index,value in data[["Defense"]][0:5].iterrows():

    print(index," : ",value)
# For example

def tuple_function():

    """ This function returns defined tuple"""

    tuple_names = ("berkant", "dogus", "kutay")

    return tuple_names



name1, name2, name3 = tuple_function()

print(name1, name2, name3)



# You can not change tuples!

tuple_numbers = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

print(tuple_numbers)

#tuple_numbers[0] = 8 # This line gives an error about assignment



# You can print some part of tuples

print(tuple_numbers[4:8])
x = 2  # x is now defined within the module namespace

def foo():

    x = 3 # x is now defined within the local namespace of function

    print(x)



foo()
x = "2"  # x is now defined within the module namespace

def example():

    x = "3" # x is now defined as 3 within the local namespace of example

    def method():

        x = "4" # x is now defined as 4 within the local namespace of method

        def function():

            x = "5" # x is now defined as 5 within the local namespace of function

            print("Function Scope: " + x)

        function()

        print("Method Scope: " + x)

    method()

    print("Example Scope: " + x)

example()

print("Module Scope: " + x)
x = "2"  # x is now defined within the module namespace

def example():

    x = "3" # x is now defined as 3 within the local namespace of example

    def method():

        global x  # x will now be defined as being within the module scope 

        x = "4" # x is now defined as 4 within the local and module namespace

        def function():

            x = "5" # x is now defined as 5 within the local namespace of function

            print("Function Scope: " + x)

        function()

        print("Method Scope: " + x)

    method()

    print("Example Scope: " + x)

example()

print("Module Scope: " + x)
print(type(dir))

print(type(print))

print(type(open), end = "\n\n")



# How can we learn what is built in scope

import builtins

print(dir(builtins))
#nested function

def square():

    """ Return square of value """

    def add():

        """ Add two local variable """

        number1 = 4

        number2 = 3

        return number1 + number2

    return add() ** 2

print(square()) 
def ask_ok(prompt, retries=4, reminder='Please try again!'):

    print("")

    while True:

        answer = input(prompt)

        if answer in ('y', 'ye', 'yes'):

            return True

        if answer in ('n', 'no', 'nop', 'nope'):

            return False

        retries = retries - 1

        if retries < 0:

            raise ValueError('Invalid user response!')

        print(reminder)



#ask_ok('Do you really want to quit?')
# Flexible arguments *args

def multiply(*args):

    z = 1

    for num in args:

        z *= num

    print(z)



multiply(4, 5)

multiply(10, 9)

multiply(2, 3, 4)

multiply(3, 5, 10, 6)



print("")



# Flexible arguments **kwargs that is dictionary

def car_info(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items(): # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, ":", value)

car_info(brand = 'ford', model = 'mustang', year = 1964)
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(9))



total = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(total(5,2,8))
list_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

list_cubes = list(map(lambda x: x**3, list_numbers))

print(list_cubes)
# Iteration example

name = "berkant"

iterable_name = iter(name)



print(next(iterable_name))    # print next iteration

print(next(iterable_name))    # print next iteration         

print(next(iterable_name))    # print next iteration

print(next(iterable_name))    # print next iteration

print(*iterable_name)         # print remaining iteration
list_rank = [1,2,3]

list_name = ["berkant","dogus","kutay"]



zip_result = zip(list_rank, list_name)

print(zip_result)

#print(type(zip_result))



print("") 



list_zip_result = list(zip_result)  #converting zip to list type

print(list_zip_result)

#print(type(list_zip_result))



print("")



iterable_zip_result = iter(list_zip_result) 

print(next(iterable_zip_result))   # print next iteration

print(*iterable_zip_result)        # print remaining iteration

#print(type(iterable_zip_result))
unzip_result = zip(*list_zip_result)

list_rank, list_name = list(unzip_result) # unzip returns tuple



print(list_rank)

print(list_name)



print(type(list_rank))

print(type(list(list_name))) #if we want to change data type tuple to list we need to use list() method.
squares = []

for x in range(10):

    squares.append(x**2)

    

print(squares)
squares = list(map(lambda x: x**2, range(10)))

print(squares)
squares = [x**2 for x in range(10)]

print(squares)
output = [(x,y) for x in [1,2,3] for y in [3,1,4] if x != y]



print(output)
output = []

for x in [1,2,3]:

    for y in [3,1,4]:

        if x != y:

            output.append((x, y))



print(output)
# Another example

result = ["Positive" if i > 0  else "Negative" if i<0 else "Zero" for i in range(-10,10,1)]

print(result)
# Lets return pokemon.csv and make one more list comprehension example

# Lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed)/len(data.Speed)

print("Threshold : ", threshold)



data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

print(data.loc[:10,["speed_level","Speed"]]) # We will learn loc more detailed later
data = pd.read_csv('../input/pokemon.csv')

data.head()  # Head shows first 5 rows
data.tail()  # Tail shows last 5 rows
# Columns gives column names of features

data.columns
# Shape gives number of rows and columns in a tuble

data.shape
# Info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
data.rename(columns={"Type 1":"type1", "Type 2":"type2"}, inplace=True)

data.columns
# To replace spaces with an underscore

data.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in data.columns]

print(data.columns)
# To replace upper case with lower case

data.columns = [column.lower() for column in data.columns]

print(data.columns)
# For example lets look frequency of pokemom types

print(data.type1.value_counts(dropna = False, sort = True, ascending = True))  # if there are nan values that also be counted

# sort : boolean, default True   =>Sort by values

# dropna : boolean, default True =>Don’t include counts of NaN.

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
# For example max Attack is 190 or min Defense is 5

# First quantile of HP is 50

# Median (Second Quantile) of Speed is 65

data.describe() #ignore null entries
print(data.columns)
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='attack',by = 'legendary')

plt.show()
data_head = data.head()

data_head
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

data_melted = pd.melt(frame = data_head, id_vars = 'name', value_vars = ['attack','defense'])

data_melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

data_melted.pivot(index = 'name', columns = 'variable', values = 'value')
# Firstly lets create 2 data frame

data_head = data.head()

data_tail = data.tail()

conc_data_row = pd.concat([data_head, data_tail], axis = 0, ignore_index = True) # axis = 0 : adds dataframes in row

conc_data_row
# Firstly lets create 2 data frame

data_attack_head = data.attack.head()

data_defense_head = data.defense.head()

conc_data_row = pd.concat([data_attack_head, data_defense_head], axis = 1)

conc_data_row
# To learn data types in dataset

data.dtypes
# lets convert object(str) to categorical and int to float.

data.type1 = data.type1.astype('category')

data.speed = data.speed.astype('float')
# As you can see type1 is converted from object to categorical

# And speed is converted from int to float

data.dtypes
# Lets look at does pokemon data have nan value

# As you can see there are 800 entries. However type2 has 414 non-null object so it has 386 null object.

data.info()
# Lets check type2

data.type2.value_counts(dropna = False)

# As you can see, there are 386 NAN value
# Lets drop nan values

data_dropna = data   # also we will use data to fill missing value so I assign it to data1 variable

data_dropna.type2.dropna(inplace = True)  # inplace = True means we do not assign it to new variable. 

# Changes automatically assigned to data

# So does it work ?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert  data.type2.notnull().all() # returns nothing because we drop nan values

data.info()
assert  data.type2.notnull().all() # returns nothing because we drop nan values

# # With assert statement we can check a lot of thing. For example

# assert data.columns[1] == 'name'

# assert data.speed.dtypes == np.float
# data frames from dictionary

brand = ["Ford","Opel"]

model = ["Focus","Corsa"]

list_label = ["Brand","Model"]



list_column = [brand, model]

data_zipped = list(zip(list_label, list_column))



data_dictionary = dict(data_zipped)

dataFrame = pd.DataFrame(data_dictionary)

dataFrame
# Add new columns

dataFrame["Year"] = ["2012","2015"]

dataFrame
# Broadcasting

dataFrame["Color"] = "White" # Broadcasting entire column

dataFrame
data_ads = data.loc[:, ["attack", "defense", "speed"]] 

data_ads.plot(subplots = True)

plt.show()
# scatter plot  

data_ads.plot(kind = "scatter", x= "attack", y = "defense")

plt.show()
# hist plot  

data_ads.plot(kind = "hist", y = "defense", bins = 50, range= (0,250), density = 1)

plt.show()
# histogram subplot with non cumulative and cumulative

figure, axes = plt.subplots(nrows = 2, ncols = 1)

data_ads.plot(kind = "hist", y = "defense", bins = 50, range= (0,250), density = 1, ax = axes[0])

data_ads.plot(kind = "hist", y = "defense", bins = 50, range= (0,250), density = 1, ax = axes[1], cumulative = True)

plt.savefig('graph.png')

plt.show()
data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# However we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data_datetime = data.head()

date_list = ["2019-06-21","2019-06-22","2019-06-23","2020-01-11","2020-01-12"]

datetime_object = pd.to_datetime(date_list)

data_datetime["date"] = datetime_object

# lets make date as index

data_datetime = data_datetime.set_index("date")

data_datetime 
# Now we can select according to our date index

print(data_datetime.loc["2019-06-22"])

print("---")

print(data_datetime.loc["2019-06-21":"2019-06-23"])
# We will use data_datetime that we create at previous part

data_datetime.resample("A").mean()
# Lets resample with month

data_datetime.resample("M").mean()

# As you can see there are a lot of nan because data_datetime does not include all months
# In real life (data is real. Not created from us like data_datetime) we can solve this problem with interpolate

# We can interpolete from first value

data_datetime.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data_datetime.resample("M").mean().interpolate("linear")
# Read csv file

data = pd.read_csv('../input/pokemon.csv')

data= data.set_index("#")

data.head()
# Indexing using square brackets

data["HP"][1]
# Using column attribute and row label

data.HP[1]
# Using loc accessor

data.loc[1,["HP"]]
# Selecting only some columns

data[["HP","Attack"]]
# Difference between selecting columns: series and dataframes

print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"HP":"Defense"] # 10 and "Defense" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"] 
# From something to end

data.loc[1:10,"Sp. Atk":] 
# Creating boolean series

boolean = data.HP > 180

data[boolean]
# Combining filters

first_filter = data.HP > 180

second_filter = data.Speed > 15

data[first_filter & second_filter]
# Filtering column based others

data.HP[data.Speed < 20]
# Plain python functions

def div(n):

    return n/2

data.HP.apply(div)
# Or we can use lambda function

data.HP.apply(lambda n : n/2)
# Defining column using other columns

data["total_power"] = data.Attack + data.Defense

data.head()
# Our index name is this:

print(data.index.name)

# Lets change it

data.index.name = "index_name"

data.head()
# Overwrite index

# If we want to modify index we need to change all of them.

data.head()

# First copy of our data to data3 then change index 

data_indexed = data.copy()

# Lets make index start from 100. It is not remarkable change but it is just example

data_indexed.index = range(100, 900, 1)

data_indexed.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data = data.set_index("#")

# also you can use 

# data.index = data["#"]
# Lets read data frame one more time to start from beginning

data = pd.read_csv('../input/pokemon.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data_index = data.set_index(["Type 1","Type 2"]) 

data_index.head(100)

# data1.loc["Fire","Flying"] # howw to use indexes
dictionary = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

dataframe = pd.DataFrame(dictionary)

dataframe
# Pivoting

dataframe.pivot(index="treatment",columns = "gender",values="response")
dataframe_index = dataframe.set_index(["treatment","gender"])

dataframe_index

# Lets unstack it
# level determines indexes

dataframe_index.unstack(level = 0)
dataframe_index.unstack(level = 1)
# change inner and outer level index position

dataframe_swap = dataframe_index.swaplevel(0, 1)

dataframe_swap
dataframe
# dataframe.pivot(index="treatment", columns = "gender", values="response")

pd.melt(dataframe, id_vars = "treatment", value_vars = ["age","response"])
# We will use dataframe

dataframe
# According to treatment take means of other features

dataframe.groupby("treatment").mean() # Mean is aggregation / reduction method

# There are other methods like sum, std,max or min
# We can only choose one of the feature

dataframe.groupby("treatment").age.max() 
# Or we can choose multiple features

dataframe.groupby("treatment")[["age","response"]].min() 
dataframe.info()

# As you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#dataframe["gender"] = dataframe["gender"].astype("category")

#dataframe["treatment"] = dataframe["treatment"].astype("category")

#dataframe.info()