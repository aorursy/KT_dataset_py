#The properties that all Dog objects must have are defined in a method called .init(). Every time a new Dog object is created, 

#.init() sets the initial state of the object by assigning the values of the object’s properties. That is, .init() initializes each new instance of

#the class.You can give .init() any number of parameters, but the first parameter will always be a variable called self. When a new class instance 

#is created, the instance is automatically passed to the self parameter in .init() so that new attributes can be defined on the object.



class Dog:

    # Class attributes are attributes that have the same value for all class instances. 

    species = "Canis familiaris"

    

    def __init__(self, name, age):  # Attributes created in .__init__() are called instance attributes. 

        self.name = name # self.name = name creates an attribute called name and assigns to it the value of the name parameter.

        self.age = age

        # All Dog objects have a name and an age, but the values for the name and age attributes will vary depending on the Dog instance.

        

# Creating a new object from a class is called instantiating an object.    

buddy = Dog("Buddy", 9)

miles = Dog("Miles", 4)

# When you instantiate a Dog object, Python creates a new instance and passes it to the first parameter of .__init__(). 

# This essentially removes the self parameter, so you only need to worry about the name and age parameters.

 

# After you create the Dog instances, you can access their instance attributes using dot notation: buddy.name buddy.species



# Although the attributes are guaranteed to exist, their values can be changed dynamically:

buddy.age = 10

miles.species = "Felis silvestris"

# An object is mutable if it can be altered dynamically. F.e., lists and dictionaries are mutable, but strings and tuples are immutable.
class Dog:

    species = "Canis familiaris"



    def __init__(self, name, age):

        self.name = name

        self.age = age



    # Instance method, .description() returns a string displaying the name and age of the dog.

    def description(self):

        return f"{self.name} is {self.age} years old"



    # Another instance method, .speak() has one parameter called sound and returns a string containing the dog’s name and the sound the dog makes.

    def speak(self, sound):

        return f"{self.name} says {sound}"

    

    # Replace .description() with __str__()

    def __str__(self):

        return f"{self.name} is {self.age} years old"

    

miles = Dog("Miles", 4)

miles.description()

miles.speak("Woof Woof")

# When you print(miles), you get a cryptic looking message telling you that miles is a Dog object at the memory address 0x00aeff70.

# You can change what gets printed by defining a special instance method called .__str__().

miles = Dog("Miles", 4)

print(miles)
class Length:



    __metric = {"mm" : 0.001, "cm" : 0.01, "m" : 1, "km" : 1000,

                "in" : 0.0254, "ft" : 0.3048, "yd" : 0.9144,

                "mi" : 1609.344 }

    

    def __init__(self, value, unit = "m" ):

        self.value = value

        self.unit = unit

    

    def Converse2Metres(self):

        return self.value * Length.__metric[self.unit]

    

    def __add__(self, other):

        l = self.Converse2Metres() + other.Converse2Metres()

        return Length(l / Length.__metric[self.unit], self.unit )

    

    def __str__(self):

        return str(self.Converse2Metres())

    

    def __repr__(self):

        return "Value and unit: " + str(self.value) + ", '" + self.unit 



if __name__ == "__main__":

    z = Length(4.5, "yd")

    print(repr(z))

    print("Variable value in meter: ",z)
import numpy as np

#NumPy is a python library used for working with arrays.

#It also has functions for working in domain of linear algebra, fourier transform, and matrices.

#We have lists that serve the purpose of arrays, but they are slow.NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.



import pandas as pd 

#Why pandas: you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — 

#a table, basically — then let you do things like:

#Calculate statistics and answer questions about the data, like: What's the average, median, max, or min of each column?

#Does column A correlate with column B?

#What does the distribution of data in column C look like?

#Clean the data by doing things like removing missing values and filtering rows or columns by some criteria

#Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.

#Store the cleaned, transformed data back into a CSV, other file or database



import os

#The OS module in python provides functions for interacting with the operating system.

#This module provides a portable way of using operating system dependent functionality.

#The *os* and *os.path* modules include many functions to interact with the file system.



import matplotlib.pyplot as plt

#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.



import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.       
data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

#data=pd.read_csv("kaggle/input/google-play-store-apps/license.txt")

#data=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")

data.head(10)
data.info()
data.Rating.plot(kind = 'line', color = 'g',label = 'Rating',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

data.head(10)
# tail shows last 5 rows

data.tail()
print(data.App.value_counts(dropna =False)) # if there are nan values that also be counted
data.describe()
data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

data.head(10)
#1. Importing the required libraries for EDA(pandas,numpy,seaborn,matplotlib)

#2.Loading the data into the data frame (just read the CSV into a data frame and pandas data frame does the job for us.)

#3. Checking the types of data

data.dtypes

# We have to convert that string to the integer data only then we can plot the data via a graph. 
import pandas as pd

d1 = {'Name': ['Pankaj', 'Meghna', 'David'], 'ID': [1, 2, 3], 'Role': ['CEO', 'CTO', 'Editor']}

source_df = pd.DataFrame(d1)

print(source_df)

print("--------")

# drop columns  

result_df = source_df.drop(columns=['ID', 'Role'])

print(result_df)

print("--------")

#Drop DataFrame Columns and Rows in place

source_df.drop(columns=['ID'], index=[0], inplace=True)

print(source_df)
#Using labels and axis to drop columns and rows

import pandas as pd

d1 = {'Name': ['Pankaj', 'Meghna', 'David'], 'ID': [1, 2, 3], 'Role': ['CEO', 'CTO', 'Editor']}

source_df = pd.DataFrame(d1)

print(source_df)

print("--------")



# drop rows

result_df = source_df.drop(labels=[0, 1], axis=0) #axis:The possible values are {0 or ‘index’, 1 or ‘columns’}, default 0.

print(result_df)

print("--------")



# drop columns

result_df = source_df.drop(labels=['ID', 'Role'], axis=1)

print(result_df)
#4. Dropping irrelevant columns

#This step is certainly needed in every EDA because sometimes there would be many columns 

#that we never use in such cases dropping is the only solution.

# Dropping irrelevant columns

data = data.drop(labels= ['Last Updated', 'Type', 'Price', 'Content Rating','Android Ver','Current Ver'], axis=1)

data.head()
#5. Renaming the columns

#Most of the column names are very confusing to read, so I just tweaked their column names.

data=data.rename(columns={'App':'Application Name'} )

data.head()
#6. Dropping the duplicate rows

#This is often a handy thing to do because a huge data set.I remove all the duplicate value from the data-set. 

#I had 10841 rows of data but after removing the duplicates 10356 data meaning that I had 485 of duplicate data.

data.shape

# Rows containing duplicate data

duplicate_rows_data = data[data.duplicated()]

print('number of duplicate rows: ', duplicate_rows_data.shape)

#Now let us remove the duplicate data because it's ok to remove them.
# Used to count the number of rows before removing the data

data.count()
#10841 rows and we are removing 485 rows of duplicate data.

# Dropping the duplicates 

data = data.drop_duplicates()

data.head(5)
data.count()
#7. Dropping the missing or null values.

# Finding the null values.

print(data.isnull().sum())

#This is the reason in the above step while counting Rating had 9367.
# Dropping the missing values.

data = data.dropna() 

data.count()

#Now we have removed all the rows which contain the Null or N/A values
# After dropping the values

print(data.isnull().sum()) 
data.dtypes
#data['Reviews'] = data['Reviews'].astype(float)

#data['Reviews'] = pd.to_numeric(data['Reviews'])

#How to Convert String to Integer in Pandas DataFrame
#8. Detecting Outliers

#Because outliers are one of the primary reasons for resulting in a less accurate model. Hence it’s a good idea to remove them.

#Often outliers can be seen with visualizations using a box plot. 

sns.boxplot(x=data['Rating'])

#Shown below are the box plot of Rating.You can find some points are outside the box they are none other than outliers.
data.boxplot(column='Rating')
Q1 = data.quantile(0.25)

Q3 = data.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
data = data[~((data < (Q1-1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

data.shape