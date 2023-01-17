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
df = pd.read_csv("/kaggle/input/fifa19/data.csv") # i called csv of data

df.head() # i showed first 5 index of data
df.columns # i looked all columns
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
plt.scatter(df.Overall,df.Potential,color='green',alpha=0.4)

plt.title('Overall - Potential')

plt.show()
df.Overall.plot(kind='hist',bins= 70, figsize=(10,10),color='blue')

plt.title('Overall')

plt.show()
df[(df['Overall']>90) & (df['Potential']>90)] # i want to look who has overall AND potential >90
df[(df['Nationality'])== "Turkey"] # i look turkish players.

filter = df["Club"]== "Galatasaray SK"

df[filter]



        
df[df["Club"]=="Galatasaray SK"]
df[(df['Club'])== "Galatasaray"]
import builtins

dir(builtins)

import os

dir(os)
def tubleEx():

    t=(1,2,3)

    return t



a,b,c=tubleEx()

print(a,b,c)
x = 5



def ck():

    x=10

    return x

print(x)

print(ck())
def square():

    def add():

        x=6

        y=3

        z=y +x

        return z

    print(add())

    return add()**2



print(square())
def ck(a,b=1,c=2):

    x = (a+b)*c

    return x



print(ck(5))

print(ck(5,4,3))
sq = lambda x: x**2

print(sq(5))



ck = lambda a,b,c: (a+b)*c



print(ck(5,5,5))
name = 'karaduman'

it = iter(name)

print(next(it))

print(*it)
del list
list1 = [1,2,3,4]

list2 = [5,6,7,8]



z = zip(list1,list2)

print(z)

z_list = list(z)



print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip)

print(un_list1)

print(un_list2)

print(type(list(un_list2)))

# Example of list comprehension

num1 = [5,10,20]

num2 = [i+1 for i in num1]

print(num2)
# Conditionals on iterable



num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]



print(num2)
threshold = sum(df.Overall)/len(df.Overall)

print(threshold)

df["OverallLevel"] = ["high" if i > threshold else "low" for i in df.Overall]



df.loc[1000:,["OverallLevel","Overall"]]
print(df["Club"].value_counts(dropna=False))
print(df["Nationality"].value_counts(dropna=False))
df.describe()
df.boxplot(column='Overall',by='International Reputation')

plt.show()
dfNew = df.head()

dfNew
melted = pd.melt(frame=dfNew,id_vars = 'Name',value_vars=['Overall','Nationality',"Club"])

melted
df1 = df.head()

df2 = df.tail()



concDataRow = pd.concat([df1,df2],axis=0,ignore_index=True)

concDataRow
df1 = df["Nationality"]

df2 = df["Club"]

concDataCol = pd.concat([df1,df2],axis=1)

concDataCol
df.dtypes
df['Club'] = df['Club'].astype('object')

df['Age'] = df['Age'].astype('float')
df.info()
df1 = df



df["Club"].dropna(inplace = True)
df['Club'].value_counts(dropna = False)
assert df['Club'].notnull().all()

print(df["Club"])
df["Club"].fillna('empty',inplace=True)
country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

print(zipped)

data_dict = dict(zipped)

data = pd.DataFrame(data_dict)

data
df = pd.read_csv("../input/fifa19/data.csv")

df
df1 = df.loc[:,["Overall","Potential","Composure"]]

df1.plot()

plt.show()
df1.plot(subplots = True)

plt.show()
df1.plot(kind="scatter",x="Overall",y="Potential",color="green")

plt.show()
df1.plot(kind="hist",y="Overall",bins = 200 , range=(40,100))

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

df1.plot(kind='hist',y="Potential",bins=200,range=(0,100),ax=axes[0]),

df1.plot(kind='hist',y="Potential",bins=200,range=(0,100),ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt.show()
df.describe()
timeList = ["1992-03-08","1992-04-12"]

print(type(timeList[1]))



datetimeObject = pd.to_datetime(timeList)

print(type(datetimeObject))


df2 = df.head()

dateList = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetimeObject = pd.to_datetime(dateList)

df2["Date"] = datetimeObject

df2= df2.set_index("Date")

df2
print(df2.loc["1993-03-16"])

print(df2.loc["1992-03-10":"1993-03-16"])
df
df = pd.read_csv("../input/fifa19/data.csv")

df = df.set_index("ID")

df.head()
df.Overall[20801]
df[["Overall","Potential"]]
boolean = df["Overall"] > 88

df[boolean]
firstFilter = df.Overall > 90

secondFilter = df.Potential > 90

df[firstFilter & secondFilter]
df.Overall[df.Potential>91]