# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data.csv')
data.info()
data.head()
data.columns
data.corr()
#correlation map

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True,linewidth=5,fmt='.2f',ax=ax)

plt.show()
data.Position.plot(kind='line',color='g',label='Position',linewidth=5,alpha=1,grid=True,linestyle='-.')

data.Streams.plot(color='r',label='Streams',linewidth=0.5,alpha=1,grid=True,linestyle=':')

plt.legend=('down right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind='scatter',x='Position',y='Streams',alpha=1,color='r')

plt.xlabel('Position')

plt.ylabel('Streams')

plt.title('Position')

plt.show()
data.Streams.plot(kind='hist',bins=40,figsize=(10,10))

plt.show()
dic = {'statistics' :'f_test','math':'addition'}

print(dic.keys())

print(dic.values())
dic['statistics']='t_test'

print(dic)

dic['python']='artificial_intelligence'

print(dic)

del dic['math']

print(dic)

print('python' in dic)

dic.clear()

print(dic)

data=pd.read_csv('../input/data.csv')
series=data['Streams']

print(type(series))

data_frame=data[['Streams']]

print(type(data_frame))
print(5==0)

print(5!=2)

print(4<4.0)

print(False or True)

print(False and True)
x=data['Streams']>9022725

data[x]
data[np.logical_and(data['Streams']>9022725,data['Position']<3)]
data[(data['Streams']>9022725) & (data['Position']<3)]
i=0

while i !=3:

    print('i is:',i)

    i+=1

print(i,' is equal to 3')

    
lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



for key,value in dic.items():

    print(key," : ",value)

print('')



for index,value in data[['Streams']][0:1].iterrows():

    print(index," : ",value)
def tuple_ex():

    t=(1,2,3)

    return t

a,b,c=tuple_ex()

print(a,b,c)
x=5

def f():

    x=3

    return x 

print(x)

print(f())
z=3

def f():

    t=z*3

    return t 

print(f())
import builtins

dir(builtins)
def square():

    def add():

        a=2

        b=2

        c=a+b

        return c

    return add()**2

print(square())
add=lambda x,y:x+y

print(add(3,4))

multi=lambda x,y,z:x*y*z

print(multi(3,4,5))
number=[3,9,4]

y=map(lambda x:x**3,number)

print(list(y))
name='messi'

it=iter(name)

print(next(it))

print(*it)
list1=[1,2,3,4]

list2=[9,8,6,0]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)
un_zip=zip(*z_list)

un_list1,un_list2=list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list1))
num1=[1,2,3]

num2=[i+1 for i in num1]

print(num2)
num1=[5,4,8]

num2=[i**2 if i==5 else i-7 if i<5 else i%2 for i in num1]

print(num2)
threshold=sum(data.Streams)/len(data.Streams)

data['Streams_level']=['high'if i > threshold else 'low' for i in data.Streams]

data.loc[:20,['Streams_level','Streams']]
data = pd.read_csv('../input/data.csv')

data.head()  # head shows first 5 rows
data.tail()
data.shape
print(data['Artist'].value_counts(dropna=False))
data.describe()
data.boxplot(column='Streams',by='Date')

plt.show()
data_new=data.head()

data_new
melted=pd.melt(frame=data_new,id_vars='Artist',value_vars=['Track Name','Date'])

melted
melted.pivot(index='Artist',columns='variable',values='value')

data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
data1=data['Track Name'].head()

data2=data['Streams'].head()

conc_data_row=pd.concat([data1,data2],axis=1)

conc_data_row
data.dtypes
data['Track Name']=data['Track Name'].astype('category')

data['Position']=data['Position'].astype('float')
data.dtypes
data.info()
data['Streams'].value_counts(dropna=False)
assert 1==1
assert  data['Streams'].notnull().all()
data["Streams"].fillna('empty',inplace = True)
team=['River Plate','Barcelona']

country=['Argentina','Spain']

list_label=['team','country']

list_col=[team,country]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df['establishment_year']=[1901,1899]

df
df['ranking']=1

df
data1=data.loc[:,['Position','Streams']]

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind='scatter',x='Streams',y='Position')

plt.show()
data1.plot(kind='hist',y='Position',bins=50,range=(0,250),normed=True)
fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind='hist',y='Position',bins=50,range=(0,250),normed=True,ax=axes[0])

data.plot(kind='hist',y='Position',bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig('grahp.png')

plt
data.describe()
time_list=['1995-07-09','1996-01-11']

print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
data2=data.head()

date_list=['1887-03-01','1887-04-02','1888-09-05','2012-05-10','2123-03-12']

datetime_object=pd.to_datetime(date_list)

data2['date']=datetime_object

data2=data2.set_index('date')

data2
print(data2.loc['1887-03-01'])

print(data2.loc['1887-03-01':'1888-09-05'])
data2.resample('A').mean()
data2.resample('M').mean()
data2.resample('M').first().interpolate('linear')
data2.resample('M').mean().interpolate('linear')
data=pd.read_csv('../input/data.csv')

data.head()
data['Date'][1]# indexing using square brackets
data.Date[1]# using column attribute and row label
data.loc[1,["Date"]]# using loc accessor
data[['Streams','Track Name']]
print(type(data['Track Name']))

print(type([data['Track Name']]))
data.loc[1:10,'Streams':'Date']
data.loc[1:10,'URL':]
data.loc[10:1:-1,'URL':'Region']#reverse
boolean=data.Streams>8022725

data[boolean]
first_filter = data.Streams > 8022725

second_filter = data.Date > '2017-03-10'

data[first_filter & second_filter]
data.Artist[data.Streams>8049063]
def module(n): #Plain

    return n%3

data.Streams.apply(module)
module=lambda x:x%3 #lambda function

data.Streams.apply(module)
threshold=sum(data.Streams)/len(data.Streams)

data["Streams_level"] = ["high" if i > threshold else "low" for i in data.Streams]

data.head()
print(data.index.name)

data.index.name='index_name'

data.head()
data = pd.read_csv('../input/data.csv')

data.head()
data1=data.set_index(['Track Name','Artist'])

data1.head()
dic = {"team":["A","B","C","D"],"country":["US","CA","CZ","GB"],"ranking":[5,10,1,2]}

df = pd.DataFrame(dic)

df
df.pivot(index="team",columns = "country",values="ranking")
df1 = df.set_index(["team","country"])

df1

# lets unstack it
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
df
melted=pd.melt(df,id_vars='team',value_vars=['country','ranking'])

melted
df
df.groupby('team').mean()
df.groupby("team").ranking.max() 
df.info()