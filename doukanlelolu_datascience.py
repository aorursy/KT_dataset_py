# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.info()
data.corr()
#correlation map
f, ax = plt.subplots(figsize=(13,13))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt = '.1f', ax=ax)
plt.show()
data.head(10)
data.columns

# Line Plot
# color = color , label=lable , linewidth = widt of line , alpha=opacity , grid = grid , linestyle = style of line
data.Speed.plot(kind='line', color='g',label='Speed', linewidth=1,alpha=1.0, grid =True, linestyle=':')
data.Defense.plot(color='r',label='Defense', linewidth=1, alpha =1.0, grid =True, linestyle='-.')
plt.legend(loc='upper right') # Legend = puts label into plot
plt.xlabel('x axis')
plt.ylabel ('y axis')
plt.title('Line Plot') 
plt.show()
plt.scatter(data.Attack,data.Defense,color='r')
plt.show()
#Scatter Plot 
# x=attack , y=defense
data.plot(kind='scatter', x='Attack', y='Defense', alpha=1.0, color='red')

plt.xlabel('Attack')
plt.ylabel('Defence')
plt.title('AttackDefense Scatter Plot')
plt.show()
#Histogram
#bins=number of bar in figure
data.Speed.plot(kind='hist', bins=50, figsize=(15,15))
plt.show()
# clf() = cleansit up again yoı can start a fresh
data.Speed.plot(kind='hist',bins=50)
plt.clf() # plot siler
#we can not see plot due to clf()
#Dic,Pandas and Logic Control
# create dictionary and  look its keys and values
#faster than  list
dictionary = {'spain':'madrid','usa':'vegas'}
print(dictionary.keys())
print(dictionary.values())
# keys  have to be immutable objects like string, boolean, float, integer or tubles
# list is not  immutable
# keys are unique
dictionary['spain']="barcelona"  # update existing entry
print(dictionary)
dictionary['france']="paris"# add  new entry
print(dictionary)
del dictionary['spain']  #remove entry with key 'spain'
print(dictionary)
print('france' in dictionary) # check include or not
dictionary.clear() # remove all entries is dict
print(dictionary)
### PANDAS
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
series = data['Defense']  #data ['Defense']=series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] =data frame
print(type(data_frame))
#comparison  operator
print(3>2)
print(3!=2)
# Boolean operators
print(True and False)
print( True or False)
#1- Filtering pandas data frame
x = data['Defense'] > 200 #there are only 3 pokemonswho have higher defense value than 200
data[x]
#2 - filtering pandas with logical_and
#There ara only 2 parameters who have higher defence value than 200 and higher attack value than 100
data[np.logical_and(data['Defense']>200, data['Attack']>100)] # iki koşulunda sağlanmasını istedim logic_and numpy kütüphanesinden çağırdım.
#This is also same with previous code line.
data[(data['Defense']>200) & (data['Attack']>100)]
# While and For Loops
# Stay in loop if condition ( i is not equal 5) is true
i=0
while i!=5:
    print('i is:',i)
    i+=1
print(i,'is equal to 5')
lis=[1,2,3,4,5]
for i in lis:
    print('i is:',i)
print('')

#Enumerate index and value of list
#index= value=0:1,1:2,2:3,3:4,4:5
for index, value in enumerate(lis):
    print(index,":",value)
print('')

# for dictionaries
dictionary = {'usa': 'vegas', 'france': 'paris'}
for key,value in dictionary.items():
    print(key,":",value)
print('')
# for pandas wecanachieve indexa and value
for index,value in data[['Attack']][0:1].iterrows():
    print(index,":",value)
### User defined fuction
def tuble_ex():
    """return defined tuble"""
    t=(1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)

#Scope
x=2
def f():
    x=3
    return x
print(x) #x=2 global scope
print(f()) # x=3 local scope
# What is there is no local scope
x=5
def f():
    y=2*x # there is no local scope x
    return y
print(f()) # it uses global scope c
#Built in scope
import builtins
dir(builtins)
#nested function
def square():
    """return square of value"""
    def add():
        """add two local variable"""
        x=2
        y=3
        z=x+y
        return z
    return add()**2
print(square())
#default flexible  arguments
#default arguments
def f (a,b=1,c=2):
    y=a+b+c
    return y
print(f(5))
print(f(5,4,3)) #change default arguments
#flexible arguments *args
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
#*kwargs that is dictionary
def f (**kwargs):
    """print key and value of dictionary"""
    for key,value in kwargs.items():
        print(key,":",value)
f(country='spain',capital='madrid',population=1321)
#lambda fuction # faster way way of writing function
square = lambda x:x**2 
print(square(4))
tot = lambda x,y,z: x+y+z
print(tot(1,2,3))
#Anonymous functions
#map(func,seq): applies a fuction to all the items in a list
number_list=[1,2,3]
y= map (lambda x:x**2,number_list)
print(list(y))
#ITERATORS
#iteration example
name="ronaldo"
it = iter(name)
print(next(it)) # print next iteration
print(*it) # print remaining iteration
#zip example
# zip(): zip list
list1=[1,2,3,4]
list2=[5,6,7,8]
z = zip(list1,list2)
print(z)
z_list=list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2= list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))
print(type(list(un_list1)))
# list comprehension
#example of list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1]
print(num2)
# conditionals on iterable
num1= [5,10,15]
num2=[i**2 if i==10 else i-5 if i <7 else i+5 for i in num1]
print(num2)
# lets return pokemon csv and make one more list omprehension example
#lets classify pokemons whether they have high or low speed. Out threshold is average speed.
threshold = sum(data.Speed)/len(data.Speed)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10,["speed_level","Speed"]] 
#CLEANING DATA
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.head() # head shows first 5 rows
#data.Type1  data da boşluk  var
data.tail()  # tail shows last 5 rows

data.columns 

data.shape

data.info()
#Exploraty Data Analysis (EDA)
#for example lets look frequency of pokemon types
print(data['Type 1'].value_counts(dropna=False)) # if there ara non values that also be counted
# as it can be seen below there are 122 water pokemon or 70 grass pokemon
# for example max HP is 255 or min defense is 5
data.describe()  # ignore null entries
#visual exploraty data analysis
#for example : compare attack of pokemons that are legandry or not
#black line at top is max
#blue line at top is %75
#red line is median %50
#blue line is at bottom is %25
#black line at bottom is min
data.boxplot(column='Attack',by='Legendary')
plt.show()
#TIDY DATA
#firstly i create new data from pokemons data to explain melt nore easily
data_new = data.head()
data_new
#lets melt
#id_vars = what we do not wish to melt
#value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars='Name',value_vars=['Attack','Defense'])
melted
#PIVOTING DATA  # reverse of melting
#index is name
# i want to make that columns ara variable
# finally values in columns are value
melted.pivot(index='Name',columns='variable',values='value')
#concatenating data
#firstly lets create 2 data frame
data1 = data.head()
data2 = data.tail()
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True) # axis=0 : adds dataframes in row
conc_data_row
data1 = data['Attack'].head()
data2=data['Defense'].head()
conc_data_col=pd.concat([data1,data2],axis=1)  #axis=1 : adds dataframes in column
conc_data_col
#DATA TYPE
# there are 5 basic data : object(string),boolean,integer,float and categorical
data.dtypes
### convert object to categorical and int to float
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
data.dtypes
#missing data and testing with assert
#lets ook at does pokemon have nan value
#as you can see there are 800 entries. however Type 2 has 414 non-null object so it has 386 null object
data.info()
#lets chech Type 2 
data['Type 2'].value_counts(dropna=False)
#as you can see, there are 386 NAN value
# lets drop nan values 
data1=data
data1["Type 2"].dropna(inplace=True) #inplace = true means we do not asign it to new variable.
# lets check with assert statement
#assert statement:
assert 1==1 #return  nothing because it is true
assert data['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace=True)
assert data['Type 2'].notnull().all() # returns nothing because we do not have nan values
## with assert statement we can check a lot of thing. for example
#assert data.columns[1]=='Name'
assert data.Speed.dtypes == np.int
#PANDAS FOUNDATION
#data frames from dictioanry
country=["Spain","France"]
population=["11","33"]
list_label=["country","population"]
list_col=[country,population]
zipped = list(zip(list_label,list_col))
data_dict=dict(zipped)
df= pd.DataFrame(data_dict)
df
#add new column
df["capital"]=["madrid","paris"]
df
#broadcasting
df["income"]=0 #broadcasting entire column
df
#Visual Exploratory Data Analysis
#plotting all data
data1=data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
plt.show()
#subplots
data1.plot(subplots=True)
plt.show()
### scatter plot
data1.plot(kind="scatter",x="Attack",y="Defense")
plt.show()
#hist plot
data1.plot(kind="hist",y="Attack",bins=50,range=(0.250),normed=True)
plt.show()
#histogram subplot wit non cumulative and cumulative
fig,axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0])
data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative =True)
plt.savefig('graph.png')
plt

#statistical exploratory data analysis
data.describe()
#indexing pandas time series
time_list=["1992-03-08","1992-04-12"]
print(type(time_list[1])) # as you can see date  is string
#however we want it to be datetime object
datetime_object= pd.to_datetime(time_list)
print(type(datetime_object))
#close warning
import warnings
warnings.filterwarnings("ignore")

data2=data.head()
date_list=["1992-01-10","1992-02-10","1992-03-23","1992-03-15","1993-03-16"]
datetime_object=pd.to_datetime(date_list)
data2["date"]= datetime_object
data2=data2.set_index("date")
data2
#now we can select according to our date index
print(data2.loc["1993-06-16"])
print(data2.loc["1992-03-10":"1993-03-15"])
#resampling pandas time series
#we will use data2 that we create at previous part
data2.resample("A").mean()

data2.resample("M").mean()
#ın real life (data is real.not created from us like data2)we cansolve this problem with interpolate
#we can interpolarte from first value
data2.resample("M").first().interpolate("linear")
#1,2,3,,5,6,7 linear interpolation aradaki değer 4
data2.resample("M").mean().interpolate("linear")
#Manipulating Data Frames with Pandas
#read data
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data=data.set_index("#")
data.head()
#indexing using square brackets
data["HP"][1]
#using column attribute and row label
data.HP[1]
#using loc accessor
data.loc[1,["HP"]]
### selecting only same columns
data[["HP","Attack"]]
#slicing data frame
print(type(data["HP"])) #series
print(type(data[["HP"]])) #data frames
#slicing andindexing series
data.loc[1:10,"HP":"Defense"]
#reverse slicing
data.loc[10:1:-1,"HP":"Defense"]
#from something to end
data.loc[1:10,"Speed":]
### filtering data frames
#creating boolean series
booelan = data.HP >200
data[booelan]
#combining filters
first_filter = data.HP >150
second_filter=data.Speed > 35
data[first_filter & second_filter]
#filtering column based others
data.HP[data.Speed<15]
#transforming  data
def div(n):
    return n/2
data.HP.apply(div)
data.HP.apply(lambda n:n/2)
#defining column using other column
data["total_power"]=data.Attack+data.Defense
data.head()
#index objects and labeled data
#out index name is this:
print(data.index.name)
#lets change it
data.index.name = "index_name"
data.head()
#overwrite index
# if we want to modify index we need to change all of them
data.head()
#first copy of our data to data3 then change index
data3 = data.copy()
#lets make index start from 100. it is not remarkable change but it is 
data3.index =range (100,900,1)
data3.head()
#hierarchical indexing
data.head()
data1=data.set_index(["Type 1","Type 2"])
data1.head(100)
#pivoting data frames
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,89,5]}
df = pd.DataFrame(dic)
df
#pivoting
df.pivot(index="treatment",columns="gender",values="response")
#stacking and unstacking dataframe
df1=df.set_index(["treatment","gender"])
df1
#level determines indexes
df1.unstack(level=0)
df1.unstack(level=1)
df2=df1.swaplevel(0,1)
df2
#melting data frames
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
#categoricals and groupby
df

df.groupby("treatment").mean() #mean is aggregation / reduction method
df.groupby("treatment").age.mean()
df.groupby("treatment")[["age","response"]].mean()
df.info()

