# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.info()
data.columns
data.head()
data.corr()
# correlation map
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = '.1f', ax = ax)
plt.show()
# Line plot
data.revenue.plot(kind='line', color='r', label='revenue', linewidth=.7, alpha=.5, grid=True, linestyle='-' )
data.budget.plot(color='g', label='budget', linewidth=.7, alpha=.8, grid=True, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')
plt.show()
# Scatter Plot
data.plot(kind='scatter', x='vote_average', y='budget', alpha=.5, color='r')
plt.xlabel('vote_average')
plt.ylabel('budget')
plt.title('Scatter Plot')
plt.show()
# Histogram
data.budget.plot(kind='hist', bins = 20, figsize = (10,10))
plt.show()
dictionary = {'usa' : 'ford', 'japan' : 'toyota', 'france' : 'renault'}
print(dictionary.keys())
print(dictionary.values())
dictionary['usa'] = "chevrolet"
print(dictionary)
dictionary['german'] = "mercedes"
print(dictionary)
del dictionary['france']
print(dictionary)
print('france' in dictionary)
dictionary.clear()
print(dictionary)

data = pd.read_csv('../input/tmdb_5000_movies.csv')
series = data['budget']
print(type(series))
data_frame = data[['budget']]
print(type(data_frame))
x = data['budget']>260000000
data[x]
data[np.logical_and(data['budget']>260000000, data['vote_average']>7)]
data[(data['budget']>260000000) & (data['vote_average']>7)]
i = 0
while i != 5 :
    print('i is: ', i)
    i += 1
print(i, 'is equal to 5')
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

for index, value in enumerate(lis):
    print(index," : ",value)
print('') 

dictionary = {'usa' : 'ford', 'japan' : 'toyota', 'france' : 'renault'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

for index,value in data[['budget']][0:1].iterrows():
    print(index," : ",value)

def tuble_ex():
    t = (1,2,3,4)
    return t
a,b,c,d = tuble_ex()
print(a,b,c,d)

# Scope - Global/Local Scope
x = 5
def f():
    x = 7
    return x
print(x)
print(f())    
# No local scope
x = 8
def f():
    y = 3*x
    return y
print(f())
# Nested Function
def square():
    def add():
        x = 3
        y = 4
        z = x + y
        return z
    return add()**2
print(square())
# Default and Flexible Arguments
# Default Argument
def f(a, b = 1, c = 4):     #if we can not write (a) firstly, there is an error!
    x = a + b + c
    return x
print(f(3))
print(f(1,4,7))
# Flexible Argument
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,4,7,10)

def f(*kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():
        print(key, " ", value)
# (!)f(country = 'usa', brand = 'chevrolet')
# Lambda Function
square = lambda x: x**2
print(square(5))
total = lambda x,y,z: x+y+z
print(total(1,3,5))
# Anonymous Function
number_list = [1,3,5]
y = map(lambda x:x**2, number_list)
print(list(y))
# Iteration
name = "gomez"
it = iter(name)
print(next(it))
print(*it)
# zip
list1 = [1,3,5,7]
list2 = [2,4,6,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list1))
print(type(un_list2))
# List Comprehension
num1 = [1,3,5]
num2 = [i+2 for i in num1]
print(num2)
# Conditionals on iterable
num1 = [8,10,12]
num2 = [i**2 if i == 10 else i-3 if i < 10 else i+8 for i in num1]
print(num2)
threshold = sum(data.vote_average)/len(data.vote_average)
print('threshold:',threshold)
data['vote_level'] = ['high' if i > threshold else 'low' for i in data.vote_average]
data.loc[:10, ['vote_level','vote_average']]
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['status'].value_counts(dropna = False))
data.describe()
data.boxplot(column = 'vote_average', by = 'status')
data_new = data.head()
data_new
melted = pd.melt(frame = data_new, id_vars = 'original_title', value_vars = ['budget', 'revenue'])
melted
melted.pivot(index = 'original_title', columns= 'variable', values = 'value') 
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True)
conc_data_row
data1 = data['budget'].head()
data2 = data['revenue'].head()
conc_data_col = pd.concat([data1, data2], axis = 1)
conc_data_col
data.dtypes
data['status'] = data['status'].astype('category')
data['vote_count'] = data['vote_count'].astype('float')
data.dtypes
data.info()
data['status'].value_counts(dropna = False)
assert 1 == 1
#assert 1 == 2
data["homepage"].fillna('empty',inplace = True)
assert  data['homepage'].notnull().all()

country = ['usa', 'japan']
population = ['35', '10']
list_label = ['country', 'population']
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df ['capital'] = ['washington','tokyo']
df
df['income'] = 0
df
data1 = data.loc[:, ['revenue','budget','popularity']]
data1.plot()
plt.show()
data1.plot(subplots = True)
plt.show()
data1.plot(kind = 'scatter', x = 'revenue', y = 'budget')
plt.show()
data1.plot(kind = 'hist', y = 'popularity', bins = 25, range = (0,250), normed = True)
plt.show()
fig, axes = plt.subplots(nrows = 2, ncols =1)
data1.plot(kind = 'hist', y = 'popularity', bins = 25, range = (0,150), normed = True, ax = axes[0])
data1.plot(kind = 'hist', y = 'popularity', bins = 25, range = (0,150), normed = True, ax = axes[1], cumulative = True)
plt.savefig('graph.png')
plt.show()
data.describe()
time_list = ['2002-03-10','2002-04-27']
print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
data2 = data.head()
date_list = ['2002-01-10', '2002-02-10', '2002-03-10', '2003-03-20', '2003-03-30']
datetime_object = pd.to_datetime(date_list)
data2['date'] = datetime_object
data2 = data2.set_index('date')
data2
print(data2.loc['2002-01-10'])
print(data2.loc['2002-01-10':'2002-03-10'])
data2.resample('A').mean()
data2.resample('M').mean()
data2.resample('M').first().interpolate('linear')
data2.resample('M').first().interpolate('linear')
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data = data.set_index('original_title')
data.head()
data['popularity'][0]
data.popularity[0]
data.loc['Avatar',['popularity']]
data[['budget', 'revenue']]
print(type(data['popularity']))
print(type(data[['popularity']]))
data.loc['Avatar':'Spectre', 'title':'vote_count']
data.loc['Spectre':'Avatar':-1, 'title':'vote_count']
data.loc['Avatar':'Tangled', 'status':]
boolean = data.budget > 260000000
data[boolean]
first_filter = data.budget > 250000000
second_filter = data.vote_average > 7
data[first_filter & second_filter]
data.budget[data.vote_average > 8]
def div(n):
    return n/2
data.budget.apply(div)
data.budget.apply(lambda n : n/2)
data['profit_rate'] = data.revenue / data.budget
data.head()
print(data.index.name)

data.index.name = 'index_name'
data.head()
data.tail(50)
data.head()

data3 = data.copy()
data3.index = range(50, 4853, 1)
data3.head()
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.head()
data1 = data.set_index(['status', 'original_title'])
data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
df.pivot(index = 'treatment', columns= 'gender', values = 'response')
df1 = df.set_index(['treatment', 'gender'])
df1
df1.unstack(level=0)

df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])

df
df.groupby("treatment").mean()
df.groupby("treatment").age.max() 

df.groupby("treatment")[["age","response"]].min() 
