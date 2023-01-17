# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.info()
data.corr()
f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()
data.head(10) # İlk 10 pokemonu görüyoruz. Genel yapıya bakıyoruz.
data.columns # Datadaki sütun bilgileri
# Line Plot yapma

# color = renk, label = label, linewidth = çizginin kalınlığı, 

# alpha = opaklık, grid = kafesli görüntü, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g', label = 'Speed', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')

data.Defense.plot(color = 'r', label = 'Defense', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# Scatter  Plot Yapma

# x = attack, y = defense

data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color = 'red')

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.title('Attack Defense Scatter Plot')

plt.show()
# Histogram

# bins = figuredeki bar sayısı

data.Speed.plot(kind = 'hist', bins = 50, figsize = (6, 6))

plt.show()
# clf() = Bu metod, plot edilen grafiği siler.

data.Speed.plot(kind = 'hist', bins = 50)

plt.clf()
# sözlük yap ve 'key' ve 'value' değerlerine bak

dictionary = {'spain' : 'madrid', 'usa' : 'Washington'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain'] = 'barcelona'

print(dictionary)

dictionary['france'] = 'paris'

print(dictionary)

del dictionary['spain']

print(dictionary)

print('france' in dictionary)

dictionary.clear()

print(dictionary)
# del dictionary - dictionary'i memory'den tamamen siler

print(dictionary)
# data = pd.read_csv('')
series = data['Defense'] # data['Defense'] = series Data'daki Defense sütununu seri olarak al

print(type(series))

data_frame = data[['Defense']] # data[['Defense']] = data frameData'nın liste içinde 'Defense' sütununu al.

print(type(data_frame))
#1 - Pandas Data Frame'ini Filtreleme

x = data['Defense']>200 # Defansı 200'den büyük olanları x'e eşitledik.

data[x]
#2 - Numpy ile Pandas Data Frame'ini Filtreleme, 'logical_and' logical_or' ile

data[np.logical_and(data['Defense']>200, data['Attack']>100)] #200 ve 100'den büyük olanlar
data[np.logical_or(data['Defense']>200, data['Attack']>179)] #200 veya 179'dan büyük olanlar
# aynı şeyi farklı şekilde yapalım

data[(data['Defense']>200) & (data['Attack']>100)]
# while içindeki condition '==' doğru ya da '!=' yanlış oldukça loopta kal.

i = 0

while i != 5:

    print('i is:', i)

    i += 1

print(i, 'is equal to 5')
lis = [1, 2, 3, 4, 5]

for i in lis:

    print('i is:', i)

print('')



for index, value in enumerate(lis):

    print(index," : ", value)

print('')



dictionary = {'spain' : 'madrid', 'france' : 'paris'}

for key, value in dictionary.items():

    print(key, " : ", value)

print('')



for index, value in data[['Attack']][0:2].iterrows():

    print(index, " : ", value)
def tuble_ex():

    """return defined t tuble"""

    t = (1, 2, 3)

    return t

a,b,c = tuble_ex()

print(a, b, c)

x,y,_ = tuble_ex()

print(x,y)
x = 2

def f():

    x=3

    return x

print(x)

print(f())

print(x)
# Global variable can be used in function if it's not defined in function.

x = 5

def f():

    y = 2*x

    return y

print(f())
import builtins

dir(builtins)
def square():

    """return square of value"""

    def add():

        """add two local variables"""

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2

print(square())
#default arguments

def f(a, b=1, c=2):

    y = a + b + c

    return y

print(f(5))

# or

print(f(5, 4, 3))
# flexible arguments

def f(*args):

    for i in args:

        print(i)

f(1)

print('')

f(1,5,3)
# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """print key and value of dict"""

    for key,value in kwargs.items():

        print(key, ":", value)

f(country = 'spain', capital = 'madrid', population = 123456)
#lambda function

square = lambda x: x**2  # x is the argument

print(square(4))



tot = lambda x,y,z: x+y+z # x,y,z are the arguments

print(tot(1,2,3))
num_list = [1,2,3]

y = map(lambda x:x**2, num_list)

print(list(y))
# iteration example



name = 'ronaldo' # string can be made an iterable object.

it = iter(name) # make string an iterable object.

print(next(it)) # print next iteration

print(*it)      # print remaining iteration
# zip example zip(list)



list1 = [1,2,3,4]

list2 = [5,6,7,8]

num_label = ["num", "numin"]

listcol = [list1, list2]

z = zip(num_label, listcol)

print(z) # z is an obejct.



z_list = list(z) # z_list is a list.

print(z_list)



z_dict = dict(z_list)

print(z_dict)



dfz = pd.DataFrame(z_dict)

dfz
# unzip example zip(*list)



un_zip = zip(*z_list) # un_zip is an object.

un_list1, un_list2 = list(un_zip) # un_listx is a tuple.

print(un_list1)

print(un_list2)

print(type(un_list2))

print(type(list(un_list2)))
num1 = [1,2,3] # num1 is an iterable object

num2 = [i+1 for i in num1] # list comprehension,

                           #iterate over an iterable with for loop

print(num2)
# conditionals on iterable

num1 = [5, 10, 15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i + 5 for i in num1]

print(num2)
# List comprehension example with Pandas.

# We set a threshold which is the average speed.



# Let's classify the pokemons according to their speeds.



threshold = sum(data.Speed)/len(data.Speed)

# data["speed_level"] diye bir feature açıyoruz ve içine koyuyoruz.

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10, ["speed_level", "Speed"]]
data.columns
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data.head() # shows first 5 rows as default

#data.tail() # shows last 5 rows as default
data.columns
data.shape
data.info()
print(data['Type 1'].value_counts(dropna=False))
data.describe() # ignore null entries 

# gives idead about the features

#  count: number of entries

#  mean: average of entries

#  std: standart deviation

#  min: minimum entry

#  25%: first quantile

#  50%: median or second quantile

#  75%: third quantile

#  max: maximum entry
data.boxplot(column='Attack', by = 'Legendary')

plt.show()
data_new = data.head() # first 5 rows are taken

data_new
# melting

# id_vars is the base column.

# value_vars are the variable columns.



melted = pd.melt(frame=data_new, id_vars = 'Name', value_vars = ['Attack', 'Defense'])

melted
melted.pivot(index='Name', columns = 'variable', values='value')
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1, data2], axis=0, ignore_index=True)

conc_data_row
data3 = data['Attack'].head()

data4 = data['Defense'].head()

conc_data_col = pd.concat([data3, data4], axis=1)

conc_data_col
data.dtypes
data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
data.dtypes
data.info()
# Check Type 2

data["Type 2"].value_counts(dropna=False)

# There are 386 missing values vor 'Type 2' column

# dropna=False count nan values.
# We are going to drop nan (missing) values

data1 = data

data1["Type 2"].dropna(inplace = True)
assert 1==1
assert data['Type 2'].notnull().all()

# Checks 'Type 2' is notnull() for all().

# Since we dropped nan values, it returns nothing so the statement is true.
data["Type 2"].fillna('empty', inplace = True)

# We added 'empty' inplace of missing values so there is no nan value.
assert data["Type 2"].notnull().all()

# Checks if there are nan values. Returns nothing because there are no nan values.
assert data.columns[1] == 'Name' # returns nothing so is true
assert data.Speed.dtypes == np.float64 # returns nothing so is true
country = ["Spain", "France"]

population = ["11", "12"]

list_label = ["country", "population"]

list_col = [country, population]

zipped = list(zip(list_label, list_col))

data_dict = dict(zipped)

dfz = pd.DataFrame(data_dict)

dfz
# We can add new columns to that dataframe

dfz["capital"] = ["madrid", "paris"]

dfz
# We can broadcast a value to all rows in a column

dfz["income"] = 0

dfz
data1 = data.loc[:, ["Attack", "Defense", "Speed"]]

data1.plot()
# make subplots

data1.plot(subplots = True)

plt.show()
# make scatter plots

data1.plot(kind = 'scatter', x = "Attack", y = 'Defense')

plt.show()
# make histogram plot

data1.plot(kind='hist', y='Defense', bins=50, range=(0,250), density = True) 

# density: normalizes y-axis

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=1)

data1.plot(kind='hist', y='Defense', bins=50, range=(0,250), density = True, ax=axes[0])

data1.plot(kind='hist', y='Defense', bins=50, range=(0,250), density = True, ax=axes[1], cumulative=True)

plt.savefig('graph.png')

plt.show()
data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # time_list is a string.We want it to be a datetime object

datetime_obj = pd.to_datetime(time_list)

print(type(datetime_obj))
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_obj = pd.to_datetime(date_list)

data2["date"] = datetime_obj

data2 = data2.set_index("date")

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()

# We take the mean(average) of years.
data2.resample("M").mean()

# We take the mean of every month in every year.

# There are NaN values since there are no values for those months.
data2.resample("M").max().interpolate("linear")

# used .max() instead of .first()

# .first() causes an error "Invalid fill method" for interpolate().
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data.head()
data = data.set_index("#")

data.head()

# We made "#" column index.
data["HP"][1]
data.HP[1]
data.loc[1, ["HP"]]
data[["HP", "Attack"]]
print(type(data["HP"])) # 1 square bracket : series

print(type(data[["HP"]])) # 2 square bracket : dataframe
data.loc[1:10, "HP":"Defense"]

# from row 1 to 10, get columns from "HP to "Defense"
data.loc[10:1:-1, "HP":"Defense"]

# from row 10 to 1, get columns from "HP" to "Defense"
data.loc[1:10, "Speed":]

# from row 1 to 10, get columns from "Speed" to the end.
boolean = data.HP > 200 # True when HP>200

data[boolean]
first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
data[data.Speed<15]

# get columns where Speed<15
data.HP[data.Speed<15]

# from the rows where the Speed<15, get "HP" column.
# define a function called div()

# apply it to "HP" column

def div(n):

    return n/2

data.HP.apply(div)
# same thing with lambda function

data.HP.apply(lambda n: n/2)
data["total_power"] = data["Attack"] + data["Defense"]

data.head()