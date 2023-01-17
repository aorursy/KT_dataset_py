# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv")
data.info()
data.head(25)
data.corr()
f, ax = plt.subplots(figsize = (25,25))
sns.heatmap(data.corr(),annot = True, linewidth = .5, fmt = '.1f',ax=ax)
plt.show()
data.head()
data.columns
data['Economy..GDP.per.Capita.'].plot(kind= 'line',color = 'r', label='GDP Per Capita', linewidth = 1,alpha = 0.9,grid=True, linestyle = ":" )
data["Health..Life.Expectancy."].plot(color = 'r',label = 'Life Expectancy',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correlation Between GDP and HLE')
plt.show()
data.plot(kind = 'scatter', x='Economy..GDP.per.Capita.', y='Health..Life.Expectancy.', alpha=0.9, color='black')
plt.xlabel('GDP')             
plt.ylabel('HLE')
plt.title('Attack Defense Scatter Plot')
plt.show()
data.Freedom.plot(kind='hist',bins = 50, figsize = (20,20))
plt.show()
dictionary = {'cat' : 'tiger', 'reptile' : 'alligator'}
print(dictionary.keys())
print(dictionary.values())
dictionary['cat'] = 'lion'
print(dictionary)
dictionary['reptile'] = 'comodo dragon'
print(dictionary)
del dictionary['cat']
print(dictionary)
print('reptile' in dictionary)
print('lion' in dictionary)
dictionary.clear()
print(dictionary)
data = pd.read_csv("../input/2017.csv")
data.columns
series = data['Family']
print(type(series))
dataFrame = data[['Family']]
print(type(dataFrame))
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
Filter = data['Economy..GDP.per.Capita.'] > 1.48
data[Filter]
data[np.logical_and(data['Economy..GDP.per.Capita.']>1.48, data['Generosity']>0.45)]
data[(data['Economy..GDP.per.Capita.']>1.48) & (data['Generosity']>0.45)]
i = 0
while i != 7:
    
    print('i is', i)
    i+=1
print('i is equal to', i)    
lis = [1,2,3,4,5,6,7,8,9]
for i in lis:
    print('i is' ,i)
print('')
# enumaration
for index,value in enumerate(lis):
    print(index, ':' , value)
for index,value in data[['Generosity']][0:1].iterrows():
    print(index,':', value)
def tuble_ex():
    """return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
x = 2 
def f():
    x = 3
    return x
print (x)
print (f())
x = 5
def f():
    y = 2*x
    return y
print(f())
import builtins
dir (builtins)
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
def f(*args):
    for i in args:
        print(i)
f(2)    
print("")
f(8,7,1,0,65)

def f(**kwargs):
    """print key and value of the dictionary"""
    for key,value in kwargs.items():
        print(key," ", value)
f( country = 'usa', capital = 'washington DC', population = 330000000)        
kare = lambda x: x**2
print(kare(225))
abc = lambda x,y,z : x+y+z
print(abc(56,48,12))
li = [1,2,3]
a = map(lambda x:x**2 , li)
print(list(a))
name = 'nicola tesla'
it = iter(name)
print(next(it))
print(*it)
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
type(z) # we have to change it to list so
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
data.columns
d = sum(data["Economy..GDP.per.Capita."]) / len(data["Economy..GDP.per.Capita."])
data["GDP Level"] = ["high" if i> d else "low" for i in data["Economy..GDP.per.Capita."]]
data.loc[:30,["Economy..GDP.per.Capita.", "GDP Level"]]
data = pd.read_csv('../input/2017.csv')
data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Freedom'].value_counts(dropna = False))
data.describe()
data.boxplot(column = 'Generosity' , by = 'Economy..GDP.per.Capita.')
data_new = data.head()    # I only take 5 rows into new data
data_new
melted = pd.melt(frame=data_new,id_vars = 'Country', value_vars= ['Freedom','Generosity'])
melted
melted.pivot(index = 'Country' , columns = 'Freedom' , values = 'value')
malted.show()
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data.dtypes
data['Country'] = data['Country'].astype('category')
data['Family'] = data['Family'].astype('object')
data.dtypes
data.info()
data['Family'].value_counts(dropna = False)
data1 = data
data1['Generosity'].dropna(inplace = True)
assert 1==1
assert 1==2
assert data['Family'].notnull().all()
data['Family'].fillna('empty', inplace = True)
assert data['Family'].notnull().all()
# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df["capital"] = ["madrid","paris"]
df
df["income"] = 0 #Broadcasting entire column
df
data1 = data.loc[:,["Trust..Government.Corruption.","Freedom","Economy..GDP.per.Capita."]]
data1.reindex()
data1.plot()
data1.plot(subplots = True)
plt.show()
data1.plot(kind = "scatter",x="Trust..Government.Corruption.",y = "Economy..GDP.per.Capita.")
plt.show()
data1.plot(kind = "hist",y = "Economy..GDP.per.Capita.",bins = 50,range= (0,100),normed = True)
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Freedom",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Freedom",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe()
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2 
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/2017.csv')

data.head()
data = data.set_index('#')
data['Freedom'][11]
data.loc[1, ['Freedom']]
data[['Freedom','Generosity']]
print(type(data["Freedom"]))     # series
print(type(data[["Freedom"]]))   # data frames
data.loc[1:10,"Family":"Generosity"]
data.loc[10:1:-1,"Family":"Generosity"] 
data.loc[1:10 ,"Family":] 
boolean = data.Family > 1.43
data[boolean]
first_filter = data.Family > 1.41
second_filter = data.Generosity > 0.20
data[first_filter & second_filter]
data.Family[data.Generosity<0.2]
def div(n):
    return n/2
data.Family.apply(div)
data.Generosity.apply(lambda n : n/2)
data["total"] = data.Family + data.Generosity
data.head()
# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])
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
df.info()