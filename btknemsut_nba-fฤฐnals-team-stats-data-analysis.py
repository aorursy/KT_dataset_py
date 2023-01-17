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

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/championsdata.csv')
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
# Line Plot
data.FGA.plot(kind = 'line', color = 'b',label = 'FGA',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
data.PTS.plot(color = 'r',label = 'PTS',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
data.plot(kind='scatter', x='FG', y='FGA' , alpha = 0.5,color = 'red')
plt.xlabel('PTS')              
plt.ylabel('FGA')
plt.title('Attack Defense Scatter Plot')
plt.show()
# Histogram
data.TRB.plot(kind = 'hist',bins = 40,figsize = (10,10))
plt.show()
# clf() = cleans it up again you can start a fresh
data.TRB.plot(kind = 'hist',bins = 50)
plt.clf()
dictionary={'italy': 'roma', 'spain': 'madrid'}
print(dictionary.keys())
print(dictionary.values())
dictionary ['italy'] = 'roma'
print(dictionary)
dictionary['spain'] = 'madrid'
print(dictionary)
dictionary ['turkey'] = 'istanbul'
print(dictionary)
del dictionary ['spain']
print(dictionary)
print('turkey' in dictionary) 
dictionary.clear()
print(dictionary)
data=pd.read_csv('../input/championsdata.csv')
series = data['TRB']
print(type(series))
data_frame = data[['TRB']]
print(type(data_frame))
#comperation operator
print(5>4)
print(5!=4)
# Boolen operators
x = data['FGA']>100    # There are only 5 team who have higher FGA value than 100
data[x]
#There are only 2 team who have higher total rebound value than 50 and higher scores value than 125
data[np.logical_and(data['TRB']>50, data['PTS']>125 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['TRB']>50) & (data['PTS']>125)]
i = 0
while i != 8 :
    print('i is: ',i)
    i +=2 
print(i,' is equal to 8')
lis = [1,2,3,4,5,6,7]
for i in lis:
    print('i is: ', i)
print('')

for index, value in enumerate(lis):
    print(index, ' : ' ,value)
print('')

dictionary={'turkey': 'istanbul', 'italy': 'roma'}
for key, value in dictionary.items():
    print(key, ' : ', value)
print('')

for index, value in data[['TRB']][0:1].iterrows():
    print(index, ' : ', value)

def tuble_ex():
    'return defined t tuble'
    t=(1,2,3)
    return t
x,y,z = tuble_ex()
print(x, y, z)
x = 5
def f():
    x=8
    return x
print(x)       # x = 5 global scope
print(f())     # x= 8 local scope

x=3
def f():
    y=x*2
    return y
print(f())

import builtins
dir(builtins)
def square():
    'return square of value'
    def add():
        'add two local variable'
        x=3
        y=9
        z=x+y
        return z
    return add()**2
print(square())
    
def f(a, b = 1 , c = 4):
    x = a + b + c
    return x
print(f(5))
# what if we want to change default arguments
print(f(4,5,6))
def f(*args):
    for i in args:
        print(i)

        
f(1)
print('')
f(1,2,3,4)


def f(**kwargs):
    '''print key and value of dictionary '''
    for key, value in kwargs.items():
        print(key, ' : ', value)
f(country = 'turkey', capital='istanbul', population = 123456)

square = lambda x : x**2
print(square(3))
tot = lambda x,y,z : x+y+z
print(tot(2,6,9))

number_list=[4,5,6]
y = map(lambda x : x**2, number_list)
print(list(y))
name = 'ronaldo '
it = iter(name)
print(next(it))
print(*it)
list1 = [1,3,5,7]
list2 = [2,4,6,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1, un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))
num1= [1,5,9]
num2= [i + 1 for i in num1 ]
print(num2)
num1 = [3,6,9]
num2 = [i**2 if i==10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
#lets classify team  whether they have strong or weak scores . Our threshold is average speed.
thresold = sum(data.PTS) / len( data.PTS )
data['team_level'] = ['strong ' if i > thresold else 'weak' for i in data.PTS ]
data.loc[:10, ['team_level' , 'PTS']]
data = pd.read_csv('../input/championsdata.csv')
data.head()
data.tail()
data.columns
data.shape
data.info()
# For example lets look frequency of teams types
print(data['Team'].value_counts(dropna =False))
data.describe()
data.boxplot(column="FGP", by="TP" )
plt.show()
data_new =data.head()
data_new

melted = pd.melt(frame= data_new, id_vars= 'Team', value_vars= ['AST', 'TRB'])
melted
melted.pivot( index= 'Team', columns= 'variable', values = 'value')
data1= data.head()
data2= data.tail()
conc_data_row= pd.concat([data1,data2], axis=0, ignore_index =True )
conc_data_row
data1 = data['FG'].head()
data2 = data['FGA'].head()
conc_data_cool= pd.concat ([data1, data2], axis= 1)
conc_data_cool
data.dtypes
# lets convert object(str) to categorical and int to float.
data['Team'] = data['Team'].astype('category')
data['FGP'] = data['FGP'].astype('float')
data.dtypes
data.info()
data['TPP '].value_counts(dropna = False)
# As you can see, there are 6 NAN value
# Lets drop nan values
data1 = data # also we will use data to fill missing value so I assign it to data1 variable
data1[ 'TPP'].dropna(inplace = True) # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
assert 2==2  #return nothing because it is true
assert  data['TPP'].notnull().all()  # returns nothing because we do not have nan values
country = ['Turkey', 'Germany', 'Italy', 'France','Denmark']

population = ['15', '18', '20', '12','14']
list_label =  ['country', 'population']
list_col = [country, population,]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df

df ['capital'] = ['istanbul', 'berlin', 'roma', 'paris', 'Copenhagen']
df
df ['income'] = 0
df
#plotting all data
data1 = data.loc [:,['FG', 'FGA', 'PTS']]
data1.plot()
data1.plot(subplots = True)
plt.show()
data1.plot(kind='scatter', x='FG', y='FGA'  )
plt.show()
data1.plot(kind='hist', y='FGA' , bins = 50 , range= (60,100), normed = True  )
plt.show()
fig,axes = plt.subplots (nrows= 2 ,ncols= 1)
data1.plot(kind= 'hist', y='FGA', bins= 50 , range= (60,100), normed = True, ax= axes[0])
data1.plot(kind= 'hist', y='FGA', bins= 50 , range= (60,100), normed = True, ax= axes[1], cumulative = True)
plt.savefig= ( 'graph.png ')
plt.show()

data.describe()
time_list= ['1996-06-29', '1996-08-12']
print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore")
data2 = data.head()
date_list= ['1980-06-29', '1980-07-29', '1980-09-20', '1980-09-22 ', '1981-01-9']
datetime_object = pd.to_datetime(date_list)
data2 ['date']= datetime_object
data2 = data2.set_index('date')
data2
print(data2.loc['1980-09-20'])
print(data2.loc['1980-09-20' : '1980-09-29'])

data2.resample('A').mean()
#Let's sample again according to the months
data2.resample('M').mean()
data2.resample('M').first().interpolate('linear')
data2.resample('M').mean().interpolate('linear')
boolean =data.FGA >100
data[boolean]
first_filter = data.FGA>90
second_filter = data.FG>50
data[first_filter & second_filter]

data.FGA[data.FG< 28]
data.head()
def div(n):
    return n/2
data.FGA.apply(div)
data.FGA.apply(lambda n: n/2)
data['total_power'] = data.FGA + data.PTS
data.head()
data1 = data.set_index(["Year","Team"]) 
data1.head(100)
#Another example
dic = {'treatment': ['B', 'B', 'A', 'A'], 'gender' : ['F', 'M', 'F', 'M'], 'response': [15, 42,19,22], 'age': [15,68,26,9]}
df = pd.DataFrame(dic)
df
df.pivot(index = 'treatment', columns = 'gender', values = 'response')
df1 = df.set_index(['treatment', 'gender'])
df1
df1.unstack(level = 0)

df1.unstack(level = 1)

df2 = df1.swaplevel(0,1)
df2
df
pd.melt(df, id_vars ='treatment', value_vars= ['age', 'response'])

df.groupby('treatment'). mean()
df.groupby('treatment'). age.max()
df.groupby('treatment')[['age', 'response']]. min()
df.info()