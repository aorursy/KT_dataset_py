# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import DataFrame

df1 = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")

df2 = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
# In this example we use the second dataset (tmdb_5000_movies.csv).

# Let's start First Look

# This line shows info on DataFrame

df2.info()
# This line shows first 10 rows of DataFrame2

df2.head(10)
# This line shows rows and columns numbers

df2.shape
# These lines remove the features we will not use.

df = df2.drop(["homepage", "keywords", "original_language", "original_title", "overview",

               "production_companies", "production_countries", "release_date", 

               "spoken_languages", "status", "tagline"], axis=1)

df.shape
# This line shows first 10 rows of removed DataFrame

df.head(10)
# This line shows last 10 rows of DataFrame

df.tail(10)
# This line describes DataFrame columns

df.columns
# This line describes index

df.index
# This line shows numbers on non-NA values

df.count()
# These lines examine the null values and their numbers

if df.isnull().values.any():   # Check whether or not any null value

    print(df.isnull().sum())   # If there is any null value prints total numbers

else:

    print("False -> This DataFrame has not any null value")
# This line shows correlation between DataFrame values

df.corr()
# These lines visualize correlation map between DataFrame values

f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# This line shows DataFrame summary statistics (".T" means transpose of "df.describe()")

df.describe().T
# LINE PLOT

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line



df.budget.plot(kind="line", color="b", label="budget", linewidth=2, alpha=0.9, grid=True, linestyle = '--')

df.revenue.plot(color="r", label="revenue", linewidth=1, alpha=0.5, grid=True, linestyle=':')

plt.legend(loc="upper right")                        # legend = puts label into plot

plt.xlabel('x axis')                                 # label = name of label

plt.ylabel('y axis')

plt.title("Line Plot (Budget & Revenue)")            # title = title of plot

plt.show()
# SCATTER PLOT 

# x = budget, y = revenue

df.plot(kind="scatter", x="budget", y="revenue", alpha=0.7, color="purple")

plt.xlabel("Budget")                                # label = name of label

plt.ylabel("Revenue")

plt.title("Budget Revenue Scatter Plot")            # title = title of plot

plt.show()
# HISTOGRAM

# bins = number of bar in figure

df.runtime.plot(kind="hist", color="orange", bins=100, figsize=(8,8))

plt.show()
# clf() = cleans it up again you can start a fresh

df.runtime.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
# This line defines a dictionary related to movie genres

dictionary = {"2233":"future", "3344":"society", "4455":"war"}
# These lines print dictionary keys and values seperately

print(dictionary.keys())

print(dictionary.values())
# These lines change the value of "2233" and print

dictionary["2233"] = "science"

print(dictionary)
# These lines add value to the dictionary and print

dictionary["5566"] = "fiction"

print(dictionary)
# These lines delete the value of "2233" and print

del dictionary["2233"]

print(dictionary)
# These lines return boolean values of "2233" and "3344"

print("2233" in dictionary)

print("3344" in dictionary)
# These lines remove all entries in dictionary and print empty dictionary

dictionary.clear()

print(dictionary)
# These lines delete entire dictionary 

# In order to run all code you need to take comment these lines

# It gives error because dictionary is completely deleted

# Or you can use try-except structure not to get an error (like below) 

try:

    del dictionary    

    print(dictionary)

except:

    print("error")
# Import DataFrame

df = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
# Define pandas Series and pandas DataFrame related to "tmdb_5000_movies.csv"

series = df['original_title']        # data['Defense'] = pandas Series

print(type(series))

data_frame = df[['original_title']]  # data[['Defense']] = pandas DataFrame

print(type(data_frame))
# This line shows first 10 rows of pandas Series

series[0:10]
# This line shows first 10 rows of pandas DataFrame

data_frame.head(10)
# 1 - Filtering pandas DataFrame

x = df['budget']>275000000 # There are only 3 movies who have higher budget value bigger than 275M.

df[x]
# 2-A Filtering pandas DataFrame with "&"

# There are only 2 movies who have higher budget value bigger than 275M. and bigger revenue value than 1B.

y = df[(df["budget"]>275000000) & (df["revenue"]>1000000000)]

y
# 2-B Filtering pandas DataFrame with "logical_and"

# These are also same with previous code lines.

# There are only 2 movies who have higher budget value bigger than 275M. and bigger revenue value than 1B.

z = df[np.logical_and(df["budget"]>275000000, df["revenue"]>1000000000)]

z
# Stay in loop if condition is true

a = 0

while a != 5:

    print('a is: ',a)

    a +=1 

print(a,' is equal to 5')   

# Stay in loop if condition is true

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

dictionary = {"2233":"future", "3344":"society", "4455":"war"}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in df[['budget']][0:1].iterrows():

    print(index," : ",value)
