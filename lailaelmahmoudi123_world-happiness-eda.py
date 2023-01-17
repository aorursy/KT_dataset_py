# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()



# Close warnings

import warnings 

warnings.filterwarnings("ignore")









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.set_option("display.max_columns",None) 

pd.set_option("display.max_rows",None)
# Read the data 

data = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data.head(10)
data.info()  # Attribute the content of the data
data.rename(columns= {"GDP per capita": "economy","Score":"Happiness_Score","Healthy life expectancy":"health", "Freedom to make life choices":"freedom",

                     "Perceptions of corruption":"trust","Social support":"family", "Country or region":"country"},inplace = True)

data.shape #Shape give number of rows and columns in a tuple
data.columns
data.describe()  # Describing data

data.head(20)
data.tail(20)
data.dtypes
# Correlation map

# Display the negative and postive correlation between variables

data.corr

f,ax = plt.subplots(figsize=(15,10))

sns.heatmap(data.corr(), annot =True, linewidth =".5", fmt =".2f")

plt.show()





#figsize - image size

#data.corr() - Display positive and negative correlation between columns

#annot=True -shows correlation rates

#linewidths - determines the thickness of the lines in between

#cmap - determines the color tones we will use

#fmt - determines precision(Number of digits after 0)

#if the correlation between the two columns is close to 1 or 1, the correlation between the two columns has a positive ratio.

#if the correlation between the two columns is close to -1 or -1, the correlation between the two columns has a negative ratio.

#If it is close to 0 or 0 there is no relationship between them.
# Cheking missing values

data.isnull()
# Indicates values not defined in our data frame



data.isnull().sum()

# Indicates sum of missing values in our data



data.isnull().sum().sum()
data[["Happiness_Score"]].isnull().head(10)
data.sort_values("Happiness_Score", ascending = False).head(10)
data.sort_values("Happiness_Score", ascending =True).head(10)
# matplotlib is a library of python programming language for plotting graphs, there are sevearl types of graphs;

 # Line plot is better when x axis is time.

 # Scatter is better when there is correlation between two variables

 # Histogram is better when we need to see distribution of numerical data.

 # Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle
 # LINE PLOT

    

 # color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line  

data.Happiness_Score.plot(kind="line", color="B", label="Happiness_Score", linewidth=1, alpha=0.5, grid=True, figsize=(12,12))    

data.economy.plot(kind="line", color="m", label="Economy", linewidth=1, alpha=0.5, grid=True)  

data.health.plot(kind="line", color="r", label="Health", linewidth=1, alpha=0.5, grid=True)

data.family.plot(kind="line", color="G", label="family", linewidth=1, alpha=0.5, grid=True)

plt.legend(loc="upper left")  # legends= Put labels into plot

plt.xlabel("x axis")          # label = name of label

plt.ylabel("y axis")

plt.title("Line plot")        # Title of the plot

plt.show()

# Subplots



data.plot(subplots = True, figsize=(12,12))

plt.show()


plt.subplot(4,2,1)

data.family.plot(kind="line", color="orange", label="family", linewidth=1, alpha=0.5, grid=True, figsize=(10,10))

data.Happiness_Score.plot(kind="line", color="green", label="family", linewidth=1, alpha=0.5, grid=True, figsize=(10,10))

plt.ylabel("family")

plt.subplot(4,2,2)

data.Generosity.plot(kind="line", color="blue", label="Generosity", linewidth=1, alpha=0.5, grid=True, linestyle=":")

plt.ylabel("generosity")

plt.subplot(4,2,3)

data.trust.plot(kind="line", color="green", label="trust", linewidth=1, alpha=0.5, grid=True, linestyle="-.")

plt.ylabel("trust")

plt.subplot(4,2,4)

data.freedom.plot(kind="line", color="red", label="freedom", linewidth=1, alpha=0.5, grid=True)

plt.ylabel("freedom")

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind="scatter", x="Happiness_Score", y="economy", alpha=0.5, color="green", grid= False, figsize=(5,5))

plt.xlabel("Happiness_Score")    # label = name of label

plt.ylabel("economy")

plt.title("Happiness Score Economy Scatter Plot") # title = title of plot

plt.show()
data.plot(kind="scatter", x="economy", y="health", alpha=0.5, color="blue",grid =False, figsize=(5,5))

plt.xlabel("economy")    # label = name of label

plt.ylabel("health")

plt.title("Economy Health Scatter Plot") # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure



data.Happiness_Score.plot(kind="hist",color="pink", bins=100, figsize=(10,10))

plt.show()
data.Happiness_Score.head(30).plot(kind="bar",color="red")

plt.show()
dictionary={"Sweden":"Stockholm", "France":"Paris"}

print(dictionary.keys())



print(dictionary.values())
dictionary["Sweden"]=  "Stockholm" # For adding items in dictionnary

print(dictionary)



dictionary["Denmark"]= "Copenhagen" # Add new entry

print(dictionary)



del dictionary["Denmark"]

print(dictionary)



print("Denmark" in dictionary) # Check include or not



dictionary.clear()  # Remove all entries in dictionary



print(dictionary)
print(type(data))  # pandas.core.frame.DataFrame

print(type(data[["Generosity"]]))  # pandas.core.frame.DataFrame

print(type(data["Generosity"])) # pandas.core.series.Series

print(type(data["Generosity"].values)) # numpy.ndarray
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering pandas data frame



x = data["Happiness_Score"]>4.0

data[x]
# 2 - Filtering pandas with logical_and

data[np.logical_and(data["Happiness_Score"]>1.3,data["economy"]>1.3)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data["family"]>1.3) & (data["economy"]>1.3)]
i = 0

while i != 5:

    

    print ("i is:" ,i)

    i+=1

    print(i," is equal to 5")
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]



for i in lis:

    print("i is: ",i)

print("") 





# Enumerate index and value of list



# index : value = 0:1 , 1:2 , 2:3 , 3:4 , 4:5



for index, value in enumerate (lis):

    print (index, ":" , value)

    print("") 

    

    

    # For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part

dictionary = dictionary= {'France': 'Paris', 'Turkey': 'Ankara'}

for key in dictionary :

    print(key)

    

for key, value in dictionary.items():

    print(key, ",", value)
x = 4

def f():

    x = 7

    return x

print(x)      # x=4 Global scope

print(f())    # x=7 Local scope

x = 3

def f():

    y = 2*x   # There is no local scope x

    return y 

print (f())   #It uses global scope x 
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
# lets classify happiness_score whether they have high or low. Our threshold is happiness_score.

threshold = sum(data.Happiness_Score)/len(data.Happiness_Score)

data["Happiness_Score_level"] = ["high" if i>threshold else "low" for i in data.Happiness_Score]

data.loc[60:90,["Happiness_Score_level","Happiness_Score"]]
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
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
# This step of cleaning the data is very imoprtant for a data scientist, so we need to diagnostic and clean data before exploring:

# we will use head, tail, columns,shape and info methods to diagnistic data



data = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data.head()   # head shows first 5 rows


data.rename(columns={"Economy (GDP per Capita)":"economy","Score":"Happiness_Score","Health (Healthy life expectancy)": "health",

                   "Trust (Perceptions of corruption)":"trust","Freedom to make life choices":"freedom","Social support":"family"},inplace=True)

data.head(4)
# tail shows 5 last rows 

data.tail()
# columns gives columns name 

data.columns
# shape gives number of rows and columns in a tuple



data.shape
data.dtypes
data.info()
# we can filter the data

(data["Happiness_Score"]>1).head(20)
data[data["Happiness_Score"]> 1]. head(15)
data.describe()
# We tidy data with melt(). Describing melt is confusing. Therefore lets make example to understand it.



# Firstly I create new data from 2019 data to explain melt more easily.

data_new = data.head(5)    # I only take 5 rows into new data

data_new
data_new.rename (columns= {"GDP per capita":"economy","Healthy life expectancy":"health","Perceptions of corruption":"trust"

                       ,"Country or region":"Country"}, inplace = True)
data_new.head(5)
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new, id_vars = "Country",value_vars=["economy","health"])

melted
# Reverse of melting.



# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index="Country", columns = "variable", values="value")
# We can concatenate two dataframe



# Firstly lets create 2 data frame

data1 = data.head()

data2 = data.tail()

v_concat = pd.concat([data1,data2],axis=0,ignore_index=True)# axis = 0 : adds dataframe

v_concat
data11 = data.freedom.head()

data22 = data.Happiness_Score.head()

h_concat = pd.concat([data1,data2], axis=0, ignore_index=True)

h_concat
data.info()
data1 = data.freedom.head(10)

data2 = data.Happiness_Score.head(10)

data3 = data.Generosity.head(10)

h_concat = pd.concat([data1,data2,data3], axis=1)

h_concat
#There are five basic data types; boleean,object (string), float, categorical,integer

data.dtypes
# We can build data from csv as we did earlier

# But we can also build data from dictionaries

    #zip() method: This function returns a list of tuples, where the i-th tuple contains the 

    # i-th element from each of the argument sequences or iterables.



#Adding new column

#Broadcasting: Create new column and assign a value to entire column







country =["Sweden","Spain"]

population =["1000","2000"]

list_labels = ["country", "population"]

list_col = [country,population]

print(list_col)



zipped = list(zip(list_labels,list_col))

print(zipped)



data_dict = dict(zipped)

print(data_dict)



df = pd.DataFrame(data_dict)

df
df["Capital"] = ["Stokholm","Madrid"]

df
df["income"] = 0

df
# Indexing data frame

  #Indexing using square brackets

   #Using column attribute and row label

   #Using loc accessor

   #Selecting only some columns
# Read the data 

data_t = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data_t.head(10)
data_t.rename(columns= {"Country or region":"Country","Score":"Happiness_Score","Social support":"Family", "Healthy life expectancy":"Health",

                        "Freedom to make life choices":"Freedom"}, inplace = True )
# Indexing using square brackets 



data_t["Happiness_Score"][1]
# Using columns attribute and row label

data_t.Happiness_Score[1]
# Using loc accessor



data_t[["Family","Freedom"]]
# Slicing the data 

   # Difference between selecting columns

      #series and data frames

    #Slicing and indexing series

    # Reverse slicing

    # from something to end
# Difference between selecting columns: series and dataframes



print(type(data_t["Family"])) # Series

print(type(data_t[["Family"]])) # Data Frames
# Slicing and indexing series

data_t.loc[1:10,"Health":"Generosity"]
# Reverse slicing



data_t.loc[10:1:-1,"Health":"Generosity"]
data_t.loc[1:10 ,"Perceptions of corruption":]
# Filtering data frames

  # Creating boolean series containing filters filtering columns based others

    

boolean = data_t["GDP per capita"]>1.31

data_t[boolean]
# Combining filters 



first_filter = data_t.Family>1.31

second_filter = data_t.Freedom>0.20

data_t[first_filter&second_filter]
# Transforming data

   #plain python fonction

    #lambda function: to apply arbitrary python function to every moment

    #Defining column using others columns


# Plain python functions

def div(n):

    return n/2

data_t["new_Happiness_Score"]=data_t["Happiness_Score"].apply(div)

data_t
# Lambda Function

data_t["new_Happiness_Score"] = data_t["Happiness_Score"].apply(lambda hp : hp/2)

data_t


# Defining column using others columns



data_t["new_total_Happiness_score"] = data_t.Family + data_t.Freedom + data_t.Generosity

data_t
# our index name is this:

print(data_t.index.name)

#lets change it

data_t.index.name = "index_name"

data_t
# Overwrite index

# if we want to modify index we need to change all of them.

data_t.head()
# first copy of our data to data2 then change index

data2 = data.copy()
# We can make one of the column as index

# It's like this

# data= data.set_index("Happiness_Score")

# also you can use 

data2.index = data2["Happiness_Score"]

data2.index = data2["freedom"]

data2.index = data2["Happiness_Score"]

data2.head()
data_t.info()
plt.figure(figsize= (15,10))

sns.barplot(x= data_t['Country'].head(10), y= data_t['Happiness_Score'].head(10))

plt.show()