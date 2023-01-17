# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plot

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# To read .csv(comma seperated values) file with Pandas

df = pd.read_csv('/kaggle/input/fifa19/data.csv')

df.head()
# To see columns of DataFrame

df.columns
# To see general info of DataFrame

df.info()
# Removing irrelevant int and float datas ('Unnamed', ID','Age','Special','Jersey Number')

df.drop(df.columns[[0,1,3,13,22]], axis= 1, inplace = True)
# correlation map

f,ax = plot.subplots(figsize = (30,30))

sns.heatmap(df.corr(), annot = True, linewidths = 1, fmt = '.3f', ax = ax)

plot.show()
# Positive Correlation (0.857)



df.Crossing.plot(kind = 'line', color = 'red', label = 'Crossing', linewidth = 1, figsize= (8,8), alpha = 0.4,grid = True, linestyle = ':')

df.Dribbling.plot(color = 'blue', label = 'Dribbling', linewidth = 1,alpha = 0.4,grid = True, linestyle = '--')

plot.legend(loc = 'upper right')

plot.xlabel('-ID-')

plot.ylabel('-VALUE-')

plot.title('<Crossing - Dribbling> Line Plot')

plot.show()
# Negative Correlation (-0.788)



df.GKDiving.plot(kind = 'line', color = 'red', label = 'GKDiving', linewidth = 1, figsize= (8,8), alpha = 0.5, grid = True, linestyle = ':')

df.BallControl.plot(color = 'blue', label = 'BallControl', linewidth = 0.5, alpha = 0.4, grid = True, linestyle = '-.')

plot.legend(loc = 'upper right')

plot.xlabel('-ID-')

plot.ylabel('-VALUE-')

plot.title('<GKDiving - BallControl> Line Plot')

plot.show()
# No/Weak Correlation (0.024)



df.Marking.plot(kind = 'line', color = 'red', label = 'Marking', linewidth = 1, figsize= (8,8), alpha = 0.5, grid = True, linestyle = ':')

df.Finishing.plot(color = 'blue', label = 'Finishing', linewidth = 1,alpha = 0.4,grid = True, linestyle = '-.')

plot.legend(loc = 'upper right')

plot.xlabel('-ID-')

plot.ylabel('-VALUE-')

plot.title('<Marking - Finishing> Line Plot')

plot.show()
# Positive Correlation (0.857)



df.plot(kind = 'scatter', x = 'Crossing', y = 'Dribbling', figsize= (8,8), alpha = 0.5, s=8, grid= True, color = 'blue')

plot.xlabel('Crossing')

plot.ylabel('Dribbling')

plot.title('Scatter Plot (Positive Correlation)')

plot.show()
# Negative Correlation (-0.788)



df.plot(kind = 'scatter', x = 'GKDiving', y = 'BallControl', figsize= (8,8), alpha = 0.5, s=8, grid= True, color = 'blue')

plot.xlabel('GK Diving')

plot.ylabel('Ball Control')

plot.title('Scatter Plot (Negative Correlation)')

plot.show()
# No/Weak Correlation (0.024)



df.plot(kind = 'scatter', x = 'Marking', y = 'Finishing', alpha = 0.5, grid= True, color = 'blue')

plot.xlabel('Marking')

plot.ylabel('Finishing')

plot.title('Scatter Plot (No/Weak Correlation)')

plot.show()
# This graphic shows that how many players have the Potential value on the horizontal axis.



df.Potential.plot(kind = 'hist', bins = 100, figsize = (10,10), grid = True)

plot.show()
# First index of "Name", "Club" and "Nation" columns



# dct = {'name': df.Name[0], 'club': df.Club[0], 'Nation': df.Nationality[0]}

# print(dct.keys())

# print(dct.values())
# First 10 indexes of "Name" and "Club" columns



# dct = {'name': df.Name[0:11], 'club': df.Club[0:11]}

# print(dct.keys())

# print(dct.values())
# Get the value of first index of "name" key



# a = dct["name"][0]

# print("Player Name is: ", a)
# If C. Ronaldo goes to Liverpool, update it on dictionary



# dct['club'][1] = "Liverpool"

# print(dct["name"][1] + " goes to " + dct["club"][1])
# To create a dictionary



dct = {'Name': 'Muslera', 'Club': 'Galatasaray', 'Nation': 'Uruguay'}

print(dct.keys())

print(dct.values())

print(type(dct))
# To get the value of any key 



a = dct["Name"]      # 1st option

b = dct.get("Club")  # 2nd option

c = dct["Nation"]

print(a, "plays for", b)

print("From", c)

print(type(dct))
# There is also a method called get() that will help in accessing an element from the dictionary.



print(dct.get("Name"))

print(dct.get("Club"))

print(dct.get("Position"))   # if the key does not exist, get() method returns none.

print(dct.get(1))   # if the key does not exist, get() method returns none.
# To create a new entry for dictionary



dct["Position"] = "Goalkeeper"

print(dct)



# While adding a value, if the key value already exists, the value gets updated otherwise a new Key with the value is added to the Dictionary.
# Update an entry



dct['Nation'] = "Turkey"

print(dct)
# Remove an entry in dictionary



# del dct['Nation']   # 1st option



dct.pop("Nation")    # 2nd option

print(dct)
# Remove all entries in dictionary



dct.clear()

print(dct)
# Deleting dictionary <completely>



# del dct

# print(dct)    # cause an error because "dct" no longer exists.
# Check if key exists



if "Name" in dct:

    print("There is a 'Name' key in this dictionary")
# Filtering Potential value equals to 70



x = df['Potential'] == 70

print(x)       # Prints all rows as 'True' or False'

df[x]          # Prints only 'True' rows 
# Filtering data with 'and' logical



df[(df['Overall']>90) & (df['Dribbling'] > 90)]  # 1st option

# df[np.logical_and(df['Overall']>90, df['Dribbling'] > 90)] # 2nd option
# Filtering data with 'or' logical



df[(df['Overall'] > 90) | (df['Dribbling'] > 90)]              # 1st option

#df[np.logical_or(df['Overall']>90, df['Dribbling'] > 90)]   # 2nd option
# Using while



i = 0

while i < 5:

    print(df.Name[i])

    i = i + 1

# Using for in a list



lst = df.index[(df['Overall']>90)] 

print(lst)



for i in lst:

    print(df.Name[i], "has", "<", df.Overall[i], ">", "Overall value")

# Enumerate index and value of a list



for index, value in enumerate(lst):

    print("index:", index,"Name:", df.Name[index], "value: ",value)
# Using 'for' in a dictionary to get keys and values



dct2 = {df.Name[0]: df.Overall[0], df.Name[1]: df.Overall[1], df.Name[2]: df.Overall[2]}

for key,value in dct2.items():

    print(key," : ",value, "(Overall)")
# To loop/iterate over Pandas data frame and do some operation on each rows.



for index,value in df[['Overall']][0:3].iterrows():

    print(index,")", df.Name[index], " : ",value)
# To calculate (square of x) + 2



def sqr_pls2(x):

    return x**2 + 2



print(sqr_pls2(5))

# To return a verb to the gerund form 



def gerund(verb):

    return verb + 'ing'



print(gerund('listen'))
# To test if two words starts with the same letter



def first_letter(word1, word2):

    if word1[0] == word2[0]: 

        return True

    else: 

        return False



print(first_letter('pandas', 'python'))   

print(first_letter('data', 'frame'))     
# It is possible to create a function without a return statement. These are called void and they return None. 



def hello(person):

        print('Hello', person + '! Welcome to the machine!')



void = hello('my son')

print(void)
# To understand which one is local or global



a = 5

def f():

    a = 1

    return a 



print(f())       # a = 1 local scope  (inside function)  ->  'local'  code)

print(a)         # a = 5 global scope (outside function) ->  'global' code)
# What happens if there is no local scope?



a = 5

def f():

    b = a + 2        # there is no local scope a

    return b



print(f())         # it uses global scope a



# Code searches the local scope first. Then it searchs the global scope. 
# How to use Inner Function?



def welcome(name, lastname):

  

  def fullname():

    return name + " " + lastname



  print("Hi " + fullname() + "! Welcome to the machine!")



welcome('Harold', 'Finch')
def outfnc(a):

    def innerfnc(a):

        return a + 2

    b = innerfnc(a)

    print(a, b)



# innerfnc(2)        # error: name 'innerfnc' is not defined

outfnc(2)
# What is the default argument?



def f(a, b, c = 3):    # c is the default argument

    d = (a + b) * c

    return d



print(f(2,4))          # a=2 , b=4 and c=3 (default)

print(f(2,4,5))        # we can change the default argument
# What is the flexible argument *args?



def f(*args):

    for i in args:

        print(i)

        

f(5)

print("-")

f(2,3,5,7)
# What is the flexible argument **kwargs?



def f(**kwargs):

    for key, value in kwargs.items():     #loop for dictionary

        print(key, ": ", value)

        

f(Name = 'Muslera', Club = 'Galatasaray', Age = 34)
# How to use lambda function?



doubler = lambda a : a * 2

print(doubler(5))



total = lambda a,b,c,d : a + b + c + d

print(total(2,7,9,14))



# print(total(2,7))   # error missing 2 arguments

# How to apply a function to all the items in a list

# We had list before named lst

print(lst)



triple_lst = map(lambda x: x * 3, lst)

print(list(triple_lst))
# define a list

myList = [4, 7, 0, 3]



# get an iterator using iter()

myIter = iter(myList)



## iterate through it using next() 



#prints 4

print(next(myIter))



#prints 7

print(next(myIter))



## next(obj) is same as obj.__next__()



#prints 0

print(myIter.__next__())



#prints 3

print(myIter.__next__())



## This will raise error, no items left

# next(myIter)
# How to return an iterator from a tuple and print each value



tpl = (df.Name[0], df.Name[1], df.Name[2])

itr = iter(tpl)



print(next(itr))

print(next(itr))

print(next(itr))
# Strings are also iterable objects, containing a sequence of characters



tpl2 = "Messi"

itr2 = iter(tpl2)



print(next(itr2))

print(next(itr2))

print(next(itr2))

print(next(itr2))

print(next(itr2))
# How to zip lists



lst1 = [0,2,4,6]

lst2 = [1,3,5,7]

zipped = zip(lst1,lst2)

print(zipped)    # it is an object

zippedlist = list(zipped)   # we have to make type list to print as list

print(zippedlist)
# How to unzip lists



unzip = zip(*zippedlist)

unList1,unList2 = list(unzip)  # unzip makes it tuple

print(unList1, unList2)

print(type(unList1))
# How to use list comprehension

# We had list before named lst

print(lst)



double_lst = [i * 2 for i in lst]

print(double_lst)

# How to use list comprehension with conditionals



lst_even = [i for i in lst if i % 2 == 1]  # to find odd numbers in lst

print(lst_even)



double_lst_even = [i for i in double_lst if i % 2 == 0]   # to find even numbers in double_lst

print(double_lst_even)
# How to use conditionals to find players who have high overall value



average = sum(df.Overall) / len(df.Overall)

print(average)



df["Level"] = ["high" if i > average else "low" for i in df.Overall]

df.loc[df.Overall >= 64, ["Overall", "Level"]]
df.head()  #  shows first 5 rows
df.tail()  #  shows last 5 rows
df.columns  # shows the columns of data frame
df.shape  # gives number of rows and columns
df.info()  # gives data type, number of sample or feature and memory usage
df.describe()  # ignores null entries
# How to find frequency of positions



df["Position"].value_counts() 
df.boxplot(column='Potential',by = 'Position', figsize= (11,7))

plot.show()
# Creating new data to show melting function easily



newdf = df.head(10)  # takes only 10 rows into new data

newdf

# How to use melt()



# id_vars = what we do not want to melt

# value_vars = what we want to melt

meltdf = pd.melt(frame=newdf,id_vars = 'Name', value_vars= ['Overall','Potential'])

meltdf
# How to use pivot() function



meltdf.pivot(index = 'Name', columns = 'variable', values='value')
# How to concat dataframes in row

# First, we need 2 dataframes



df1 = df.head(3)

df2 = df.tail(3)



concatdf_row = pd.concat([df1,df2],axis =0, ignore_index = True) # axis = 0 - To add dataframes in row

concatdf_row

# How to concat dataframes in column



df3 = df['Overall'].head()

df4 = df['Potential'].head()

df5 = df['Name'].head()

concatdf_col = pd.concat([df5,df3,df4],axis =1) # axis = 1 - To add dataframes in column

concatdf_col
# How to convert some data types



# integer to float



df['Potential'] = df['Potential'].astype('float')



# object to categorial



df['Position'] = df['Position'].astype('category')



df.info()
# How to detect Nan values



df['Weight'].value_counts(dropna = False) 



# There are 48 NaN values
# How to drop Nan values



df['Weight'].dropna(inplace = True)
# How to check if dropna works



assert 1==1  # if it is true returns nothing

assert 1==2  # if it is false returns AssertionError as below
assert  df['Weight'].notnull().all()  # returns nothing because we dropped Nan values
assert  df['Weight'].isnull().all()   # returns error because we dont have any Nan values
# How to check if first column of data called 'Name' 

assert df.columns[0] == 'Name'