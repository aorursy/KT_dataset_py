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
df = pd.read_csv('../input/world-happiness/2017.csv')
df.info()
df = df.rename(columns={

    "Happiness.Rank": "Happiness_Rank", "Happiness.Score": "Happiness_Score", 

    "Whisker.high": "Whisker_high", "Whisker.low": "Whisker_low", 

    "Economy..GDP.per.Capita.": "Economy_GDP_per_Capita",

    "Health..Life.Expectancy.": "Health_Life_Expectancy",

    "Trust..Government.Corruption.": "Trust_Government_Corruption",

    "Dystopia.Residual": "Dystopia_Residual"

})
df.info()
df.corr()
f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df.corr(), annot=True, linewidth=5, fmt='.1f', ax=ax)

plt.show()
df.head()
df.columns
# Line Plot

df.Economy_GDP_per_Capita.plot(kind='line', color='r', label='GDP_per_Capita', linewidth=1.3, alpha=0.7, grid=True, linestyle=':')

df.Freedom.plot(kind='line', color='b', label='Freedom', linewidth=1.3, alpha=0.7, grid=True, linestyle='--')

plt.legend(loc='upper right')

plt.xlabel('Countries')

plt.ylabel('GDP')

plt.title('Line Plot of the Countries')

plt.show()
# Let's do Scatter plot

df.plot(kind='scatter', x='Economy_GDP_per_Capita', y='Generosity', alpha=0.5, color='g')

plt.xlabel('GDP per Capita')

plt.ylabel('Generosity')

plt.title('GDP per Capita x  Generosity Scatter Plot')

plt.show()
# Let's plot the Histogram

# bins = number of bars

df.Health_Life_Expectancy.plot(kind='hist', bins=30, figsize=(7, 7))

plt.xlabel('GDP per Capita')

plt.show()
# Let's make a Dictionary

dict1 = {'turkey' : 'try', 'usa' : 'usd', 'germany' : 'eur', 'uk' : 'gbp', 'denmark' : 'dkk', 'sweden' : 'sek',

       'norway' : 'nok', 'japan' : 'jpn'}

print(dict1.keys())

print(dict1.values())
dict1['germany'] = 'euro'

print(dict1)
dict1['turkey'] = 'lira'

print(dict1)
del dict1['usa']

print(dict1)
print('denmark' in dict1)
dict1.clear()

print(dict1)
del dict1

#print(dict1)
series = df['Economy_GDP_per_Capita'] # We take the 'Country' column as series

print(type(series))
df2 = df[['Economy_GDP_per_Capita']] # We take the 'Country' column as dataframe

print(type(df2))
#1 Filtering the dataframe

x = df['Economy_GDP_per_Capita'] > 1.4

df[x]
#2 Filtering the Pandas dataframe using Numpy

df[np.logical_and(df['Economy_GDP_per_Capita']>1.4, df['Freedom']>.6)]
df[np.logical_or(df['Happiness_Score']>6.8, df['Economy_GDP_per_Capita']>1.7)]
df[(df['Economy_GDP_per_Capita']>1.5) & (df['Trust_Government_Corruption']<0.4)]
i = 0

while i !=11:

    print('{}^2 is: {}'.format(i, i**2))

    i = i+1

print(i-1, "is reached")
lis = [1, 2, 3, 4, 5, 6, 7]

for i in lis:

    print('i is: ', i)
for index, value in enumerate(lis):

    print(index, ":", value)
dict2 = {'turkey' : 'try', 'usa' : 'usd', 'germany' : 'eur', 'uk' : 'gbp'}

for key, value in dict2.items():

    print(key, ':', value)
for index, value in df[['Freedom']][0:5].iterrows():

    print(index, ":", value)
def tuble_ex():

    """return defined t tuble"""

    t = (1, 2, 3)

    return t



a, b, c = tuble_ex()

print(a,b,c)

x,y,_ = tuble_ex()

print(x,y)
x = 2



def f():

    x=3

    return 3

print(x)

print(f())

print(x)
x = 5

def a():

    y = x**2

    return y

print(a())
def square():

    """return square of value"""

    def add():

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2

print(square())
# default arguments

def f(a, b=1, c=2):

    y = a + b + c

    return y

print(f(5))

#or

print(f(5,4,3))
#flexible arquments

def f(*args):

    for i in args:

        print(i)

f(1)

print('')

f(1,5,3)
# flexible arguments **kwargs that is dictionary



def f(**kwargs):

    """print key and value of dict"""

    for key, value in kwargs.items():

        print(key, ":", value)

f(country = 'spain', capital = 'madrid', population = 123456)

#lambda function

square = lambda x: x**2 # x is the argument

print(square(4))



tot = lambda x,y,z: x+y+z

print(tot(1,2,3))
num_list = [1,2,3]

y = map(lambda x:x**2, num_list)

print(list(y))
#iteration example



name = 'ronaldo' # string can be made an iterable object.

it = iter(name)  # make string an iterable object.

print(next(it))  # print next iteration

print(*it)       # print remaining iteration      
# zip example zip(list)



list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1, list2)

print(z) #z is an abject

z_list = list(z) # z_list is an object

print(z_list)

# unzip example zip(*list)



un_zip = zip(*z_list)

un_list1, un_list2 = list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list2))

print(type(list(un_list2)))
num1 = [1,2,3] # num1 is an iterable object

num2 = [i+1 for i in num1] # list comprehension,

                           # iterate over an iterable with for loop

print(num2)
# conditionals on iterable



num1 = [5, 10, 15]

num2 = [i**2 if i == 10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
# List comprehension example with Pandas.

# We set a threshold which is the average speed.

# Let's classify the countries according to their Happiness Scores.



threshold = sum(df.Generosity)/len(df.Generosity)



#we add a ["Happiness_Level"] feature



df["Generosity_Level"] = ["high" if i > threshold else "low" for i in df.Generosity]

df.loc[:10, ["Generosity_Level", "Generosity"]]

df.columns
df.head()
df.columns
df.shape
df.info()
print(df['Country'].value_counts(dropna=False))
df.describe()
# df.boxplot(column='Happiness_Score', by = 'Family')

# plt.show()

# takes too much time to plot since it has to many values so I took it in a comment.
df_new = df.head()

df_new
#melting

# id_vars is the base column.

# value_vars are the variable columns.



melted = pd.melt(frame=df_new, id_vars = 'Country', value_vars = ['Happiness_Score', 'Freedom'])

melted
melted.pivot(index = 'Country', columns = 'variable', values = 'value')
data1 = df.head()

data2 = df.tail()

conc_data_row = pd.concat([data1, data2], axis=0, ignore_index=True)

conc_data_row
data3 = df['Economy_GDP_per_Capita'].head()

data4 = df['Generosity'].head()

data5 = df['Freedom'].head()

conc_data_col = pd.concat([data3, data4, data5], axis=1)

conc_data_col
df.dtypes
df['Happiness_Rank'] = df['Happiness_Rank'].astype('float')
df.dtypes
df.info()
df["Country"].value_counts(dropna=False)
assert df['Freedom'].notnull().all()

# it returns nothing since,

# Freedom is notnull for all rows.
# df['Freedom'].fillna('empty', inplace = True)

# We could fill any blank rows with 'empty' if there were any.
assert df.columns[1] == 'Happiness_Rank'
assert df.Happiness_Rank.dtypes == np.float64

#returns nothing since the statement is true

df.Happiness_Rank.dtypes

#returns dtype
team = ["FC Bayern", "Arsenal"]

country = ["Germany", "UK"]

list_label = ["team", "country"]

list_col = [team, country]

zipped = list(zip(list_label, list_col))

data_dict = dict(zipped)

dfz = pd.DataFrame(data_dict)

dfz
# We can broadcast a value to all rows in a column

dfz["league"] = "Europa"

dfz
data1 = df.loc[:, ["Happiness_Score", "Economy_GDP_per_Capita", "Trust_Government_Corruption"]]

data1.plot()

plt.show()
# make subplots

data1.plot(subplots = True)

plt.show()
# make scatter plots

data1.plot(kind='scatter', x = 'Economy_GDP_per_Capita', y = 'Trust_Government_Corruption')

plt.show()
# make histogram plot

data1.plot(kind='hist', y='Economy_GDP_per_Capita', bins=30, range=(0, 2), density = True)

# density: normalizes histogram

plt.show()
fig, axes = plt.subplots(nrows = 2, ncols = 1)

data1.plot(kind='hist', y='Economy_GDP_per_Capita', bins=30, range=(0 ,2), density = True, ax=axes[0])

data1.plot(kind='hist', y='Economy_GDP_per_Capita', bins=30, range=(0 ,2), density = True, ax=axes[1], cumulative = True)

# cumulative converges to 1 since it goes by summing all the previous values.

plt.show()
df.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # time_list is a string. We want it to be a datetime object

datetime_obj = pd.to_datetime(time_list)

print(type(datetime_obj))
data2 = df.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

date_time_obj = pd.to_datetime(date_list)

data2["date"] = date_time_obj

data2 = data2.set_index("date")

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()

# We take the mean (average) of years
data2.resample("M").mean()

# We take the mean of every month in every year

# There are NaN values since there are no values for those months.
data2.resample("M").first().interpolate("linear")

# We begin from "Iceland" and go to "Switzerland", and fill the missing values linearly.
data2.resample("M").mean().interpolate("linear")

# We take the mean of every month individually, put those values in.

# And then we begin from "Iceland" and go to "Switzerland", and fill the missing values linearly.