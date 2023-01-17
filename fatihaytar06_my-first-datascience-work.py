# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
data = pd.read_csv("../input/tmdb_5000_movies.csv")



data.info()
x = data.drop(['genres','keywords','production_companies','production_countries','homepage','spoken_languages'],axis = 1)
x.columns
x[(x['vote_average']>7.5) & (x['popularity']>100.0)]
data.plot(kind = 'scatter', x  = 'budget' , y = 'revenue' , color = 'red' , alpha = 0.6)

plt.xlabel('Budget')

plt.ylabel('Revenue')

plt.show()

data.plot(kind='hist', y = 'vote_average' , bins = 100,figsize=(11,11))

plt.ylabel('Vote Average')

plt.title('Movies')

plt.show()
meals = {'Spain' : 'Paella','China':'Peking_Duck','Belgium':'Moules_frites','Brasil':'Feijoada','Denmark':'Frikadeller','Turkey':'Kebab'}

print(meals.keys())

print(meals.values())
meals['France'] = 'Crepe' #update dictionary

print(meals)

del meals['Belgium'] # # remove entry with key 'Belgium'

print(meals)

print('Turkey' in meals) # check it in dictionary

meals.clear() # delete dictionary

print(meals)

i = 0

liste = ['Ali','Sinan','Hakan','Elif']

while i != 4:

    for each in liste:

        print(each)

    print('')

    i+=1

    

for index,value in enumerate(liste):

    print(index," : ",value)

print('')



meals = {'Spain' : 'Paella','China':'Peking_Duck','Belgium':'Moules_frites','Brasil':'Feijoada','Denmark':'Frikadeller','Turkey':'Kebab'}

for key,value in meals.items():

    print(key," : ",value)

print('')



for index,value in data[['title']][0:2].iterrows():

    print(index," : ",value)
a = 5

def f():

    a = 4

    return a**2

print(a) # a=5

print(f()) # a=4
#What if there is no local scope



x = 7   #global scope

def f():

    return x*9+15

print(f())
import builtins

#features of built in scope

dir(builtins)
def circleArea(r):

    

    def add(pi = 3.14):

        return pi

    return 2*add()*r

print(circleArea(4))
#default arguments

def f(a,b,c=7,d=9):

    return a*b+(c*d)

print(f(6,9))

#if we want to change default arguments

print(f(7,6,9,11))

#flexible arguments

def f(*args):

    m=0

    for i in args:

       m += i

    return m



print('m = ',f(5,6,8,3,26,76))

def f(**kwargs):

    for key,value in kwargs.items():

        print(key," : " , value)

f(country = 'Turkey' , population ='80 million',capital = 'Ankara'  )
#Lambda function

circleArea = lambda r : 2*3.14*r

print(circleArea(5))
#Iterators



list1=['London','LA','Istanbul','Paris']

iterator = iter(list1)

print(next(iterator))

print(*iterator)
#example of list comprehension

num1 = [3,5,8]

num2 = [i**2 for i in num1]

print(num2)

#conditionals on iterable

num1=[5,15,25]

num2=[i**2 if i==5 else i*9 if i==15 else i*10-15 for i in num1]

print(num2)
##Let's make an object that shows us runtime level of movies

thresold = (int)(data.runtime.mean()) #average of runtime

data['Runtime_level'] = ["Long" if i>thresold else "short" for i in data.runtime]

data.loc[:15,['Runtime_level','runtime']]
df = pd.read_csv("../input/tmdb_5000_movies.csv")

df.head()
df.tail()
df.columns
print('Data Shape :',df.shape)
df.info()
df.describe() #that shows us numerical columns



#%25 means first quantile 

#%50 means median and second quantile

#%75 means third quantile

#'mean' means average value
print(df['runtime'].value_counts(dropna=False)) #print if there are nan values that also be counted
# Box plots: visualize basic statistics like outliers, min/max or quantiles

# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

#df.boxplot(column='vote_average',by = 'vote_count')
data_new = data.head()

data_new
melted = pd.melt(frame=data_new,id_vars='original_title',value_vars=['vote_average','vote_count'])

melted
#PIVOTING DATA(Reverse of melting)

# Index is original_title

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'original_title',columns='variable',values='value')
data1=df.head()

data2=df.tail()

conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row.drop(['genres','production_companies','production_countries','keywords','spoken_languages'],axis=1)
data3 = df.head()

data4 = df.tail()

conc_data_col = pd.concat([data3,data4],axis = 1)

conc_data_col.drop(['genres','production_companies','production_countries','keywords','spoken_languages'],axis=1)
df.dtypes
df['title'] = df['title'].astype('category') #convert title from string to category

df['vote_count'] = df['vote_count'].astype('float')
df.dtypes
df.info()
df["tagline"].value_counts(dropna=False)
# Lets drop nan values

data1 = df

data1['tagline'].dropna(inplace=True)# inplace = True means we do not assign it to new variable. Changes automatically assigned to data

#so does it work?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert  data1['tagline'].notnull().all() # returns nothing because we drop nan values
data['tagline'].fillna('empty',inplace=True)
assert data1['tagline'].notnull().all() # returns nothing because we don't have nan values
#With assert statement we can check a lot of thing. For example

#assert data.columns[1] == 'Name'

#assert data.Speed.dtypes == np.int
#data frames from dictionary

country = ['Turkey','France','UK']

population = ['123','432','543']

list_label = ['country','population']

list_col =[country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df= pd.DataFrame(data_dict)

df
#add new columns

df['capital'] = ['Ankara','Paris','London']

df
#Broadcasting

df['income'] = 0 #Broadcasting the entire column(fill the entire column with 0)

df
data.columns
#Plotting all data

data1=data.loc[:,['vote_average','vote_count','budget','revenue']]

data1.plot()

plt.show()

#it seems complicated
#subplots

data1.plot(subplots=True)

plt.show()
data1.plot(kind='scatter',x='budget',y='revenue',alpha=0.8)

plt.show()
data1.plot(kind='hist',y='vote_count',bins=50,range=(0,10))

plt.ylabel('Vote Count')

plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind='hist',y='vote_count',bins=30,range=(0,10),normed=True,ax=axes[0])

data1.plot(kind='hist',y='vote_count',bins=30,range=(0,10),normed=True,ax=axes[1],cumulative=True)

plt.savefig('asd.png')

plt.show()
time_list = ['1993-02-14','1993-02-16','1994-01-13','1998-08-06']

print(type(time_list[1]))

#as we can see type of that is string

#Let's convert it to datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
#close warning

import warnings

warnings.filterwarnings('ignore')

# In order to practice lets take head of movies data and add it a time list

data2= data.head(4)

time_list2 = ['2009-12-18','2009-05-25','2012-11-06','2012-07-27']

datetime_object = pd.to_datetime(time_list2)

data2["date"] = datetime_object

#lets make date as index

data2=data2.set_index("date")

data2

print(data2['vote_average'].loc['2012-11-06'])

print(data2['vote_average'].loc['2009-05-25':'2012-07-27'])
data2.resample('A').mean()
data2.resample("M").mean() #resample with month

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
data2.resample('M').mean().interpolate('linear')
data=pd.read_csv('../input/tmdb_5000_movies.csv')

data.head()
data['vote_average'][2]
data.vote_average[4]
data.loc[2,'budget']
data[['budget','vote_count']]
print(type(data['budget'])) # series

print(type(data[['vote_count']])) #data frames
print(data.loc[:5,'vote_average':'vote_count'])
term1= data['vote_average']> 9.0

data[term1]
term2 = data.budget > 10000000

term3 = data.vote_count> 10000

data[term2 & term3]
data.budget[data.vote_average>8.0]
def div(n):

    return n/3

data.runtime.apply(div)
data.runtime.apply(lambda x : x/4)
data['total_profit'] = data.revenue - data.budget

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,4903,1)

data3.head()
# We can make one of the column as index

# It was like this

# data= data.set_index("#")

# also we can use 

# data.index = data["#"]
data = pd.read_csv('../input/tmdb_5000_movies.csv')

data.head()
data1 = data.set_index(['vote_average','runtime'])

data1.head(100)
dic = {'Name' : ['Fatih','Ali','Zehra','Ayça'],'Gender' : ['M','M','F','F'],'Age' : [21,23,26,27],'Married' : ['Y','N','N','Y'],'Salary' : [3000,2500,2700,2000]}

df = pd.DataFrame(dic)

df
#pivoting

df.pivot(index='Gender',columns='Married',values = 'Salary')

df1 = df.set_index(['Gender','Married'])

df1

#let's unstack it
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
df
#df.pivot(index='Gender',columns='Married',values = 'Salary')

pd.melt(df,id_vars='Gender',value_vars=['Salary','Married'])
df
# according to Gender take means of other features

df.groupby('Gender').mean()
# we can only choose one of the feature

df.groupby('Gender').Salary.max()
# Or we can choose multiple features

df.groupby('Gender')[['Salary','Age']].min()
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["Gender"] = df["Gender"].astype("category")

#df["Married"] = df["Married"].astype("category")

#df.info()