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
data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(), annot=True, linewidths=.5,ax=ax)

plt.show()
data.head(15)
data.columns 
#Line plot using

data.Height.plot(kind = 'line', color = 'g',label = 'Height',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')

data.Weight.plot(color='r', label='Weight', linewidth=1, alpha=0.5, grid=True, linestyle =':')

plt.legend(loc='lower left') #legend = puts to label in plot

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Weight-Heigt Line Plot')

plt.show()
#Scatter plot using

data.plot(kind='scatter',x='Height',y='Weight',color='b',alpha=0.3)

plt.xlabel('Height')

plt.ylabel('Weight')

plt.title('Height-Weight Scatter Plot')     

plt.show()
#Histogram plot using

data.Height.plot(kind = 'hist',bins = 25,figsize = (8,8),color='r')

plt.title('Height Histogram Plot')

plt.show()
#create dictionary

dictionary = {'name':'Mertcan','name2':'Sena'} #Keys Names should not same.

print(dictionary.keys())

print(dictionary.values())
dictionary['name']='Mertcan2' #update keys

print(dictionary)

dictionary['name3']='Canmert' #add new entry

print(dictionary)

del dictionary['name3'] #remove entry

print(dictionary)

print('name' in dictionary) #check in 

dictionary.clear() #remove all entry in dictionary

print(dictionary)
data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
x=data['Height']>185 #comparison operator

values=data[x]

values.head(15)
x=data[np.logical_and(data['Weight']>90, data['Height']>180 )] #logical

x.head(10)
y=data[(data['Weight']>90) & (data['Height']>180)] #boolean operator

y.head(10)
#while loop

i=0

while i != 8:

    print('i is: ',i)

    i+=1

print(i,'is equal to 8')
#for loop

list1=[1,2,3,5,4,8]

for i in list1:

    print(i)
for index,values in enumerate(list1):

    print(index,'-->',values)

    
dictionary = {'name':'Mertcan','name2':'Sena'}

for key,values in dictionary.items():

    print(key,'-->',values)
for index,values in data[['Height']][0:1].iterrows():

    print(index,'-->',values)
def tuble():

    t=(1,2,3)

    return t

a,b,c=tuble()

print(b)
x=3 #global scope

def f():

    x=5

    return x #local scope

print(x)

print(f())



def f(x):

    x=x+5

    return x

print(f(7))
def square():

    def add():

        x=3

        y=5

        z=x+y

        return z

    return add()**2

print(square())
def f(a,b=1,c=1):

    y=a+b+c

    return y

print(f(5))# if not writing b and c,inside function is valid.

print(f(5,2,3)) #if writing value inside to f of b and c, b and c changed

#flexible --> *args

def f(*args):

    for i in args:

        print(i)

f(1)

print('-------')

f(1,2,3,4,5)



#flexible --> **kwargs that is dictionary

def f(**kwargs):

    for key,value in kwargs.items():

        print(key,':',value)

f(name1='Mertcan',name2='Sena')
square=lambda x: x**2

print(square(5))

total=lambda x,y: x+y+y

print(total(5,2))
number_list=[1,2,3]

y=map(lambda x:x**2,number_list)

print(list(y))
num1=[1,2,3]

num2=[i+1 for i in num1]

print(list(num2))
#example2

num1=[10,20,30]

num2=[i**2 if i == 30 else i-10 if i < 20 else i+5 for i in num1]

print(num2)



#example3

num3=[5,15,25,35,45]

num4=[i**2 if i==5 else i+5 if i<20 else i-10 for i in num3]

print(num4)
#example with data



#threshold=sum(data.Height)/len(data.Height) -->sum function is not working. I don't know why because of that i used to syntax1

threshold=180 #syntax1

data['heigh_level']=['high' if i > threshold else 'low' for i in data.Height]

data.loc[:10,['heigh_level','Heigh']] #I don't know, Why It didn't take heigh values? Please can you help me?
data.shape
data.columns
data.info
print(data.Name.value_counts(dropna=False)) #Dropna--> not including nan

#value_counts--> how many people with the same name? 
data.describe() #Returns Quantile values 
data2=data.head(200)

data2.boxplot(column='Height',by='Year')
data_new=data.head(10)

data_new
melt1=pd.melt(frame=data_new,id_vars='Name',value_vars=['Height','Weight']) 

melt1
melt1.pivot(index = 'Name', columns = 'variable',values='value')

#I don't solving this error. When I searching to find this error cause from same values to valueError. Can you help me please?
data1=data.head(10)

data2=data.tail(5)

conca_data_r=pd.concat([data1,data2],axis=0,ignore_index=True)

conca_data_r
data1=data['Height'].head(10)

data2=data['Weight'].head(10)

conca_data_c=pd.concat([data1,data2],axis=1)

conca_data_c

data.dtypes
data['Sport']= data['Sport'].astype('category')

data.dtypes
data.Sport.value_counts(dropna=False)
data1=data

data1.Weight.dropna(inplace=True)

data

assert data.Weight.notnull().all()
data.Weight.fillna('empty',inplace=True)
assert data.Weight.notnull().all()
country=["Turkey","Spain"]

population=["9","11"]

list_title=["Country","Population"]

list_column=[country,population]

zipped=list(zip(list_title,list_column))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["Capital"]=["Ankara","Madrid"]

df
#plot

data1=data.loc[:,['Age','Height','Weight']]

data1.plot()

plt.show()
data1.plot(subplots=True)

plt.show()
#scatter

data1.plot(kind="scatter",x="Height",y="Weight")

plt.show()
#histogram

data1.plot(kind="hist", y="Age" , bins=10 , range=(0,80) , normed=True)

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist", y="Age" , bins=10 , range=(0,80) , normed=True, ax= axes[0])

data1.plot(kind="hist", y="Age" , bins=10 , range=(0,80) , normed=True, ax= axes[1], cumulative=True)

plt.show()
time_list=["1998-02-15","1998-01-15"]

print(type(time_list))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
#close warning

import warnings

warnings.filterwarnings("ignore")



data2=data.head()

date_list=["1998-01-15","1998-02-15","1993-07-17","1993-08-02","1993-08-05"]

datetime_object=pd.to_datetime(date_list)

data2["Date"]=datetime_object

#let's make index to date columns

data2=data2.set_index("Date")

data2
print(data2.loc["1998-02-15"])

print(data2.loc["1998-02-15":"1993-08-05"]) #why did it say to "Empty DataFrame"? I don't understand.
data2.resample("M").mean # By Year