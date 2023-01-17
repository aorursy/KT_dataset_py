# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/world-happiness/2017.csv")

data.columns
data.info()
import matplotlib.pyplot as plt
import seaborn as sns 
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data['Happiness.Score']
data['Happiness.Score'].plot(kind='line',color='b',label='Happiness.Score',linewidth=1)
data.Freedom.plot(kind='line',color='r',label='Freedom',linewidth=1)
plt.legend()
plt.xlabel('HS')
plt.ylabel('F')
plt.title('Happiness Score-Freedom')
plt.show()
data.plot(kind='scatter',x='Economy..GDP.per.Capita.',y='Health..Life.Expectancy.',color='g',alpha=0.6)
plt.xlabel('Economy')
plt.ylabel('HealthLife')
plt.show()
country=data['Country']
print(type(country))
dataFrame=data[['Country']]
print(type(dataFrame))
#data.head()
data[(data['Happiness.Score']>5.5)]
for index,value in data [['Whisker.high']][0:1].iterrows():
    print(index,"-",value)
i=5 #global
def f(a=7):
    i=4 #local
    b=a*i
    return b
print (f())

def cube(x):
    def a():
        x=5
        return x
    return a()**3
print(cube(8)) #because x=5 in func()
def func(x,y,z=5):
    q=x*y*z
    return q
print(func(2,3))

def func(*args):
    for i in args:
        print(i)
func(5,8,9)
#Lambda Function
func=lambda x,y,z=5:x*y*z
print(func(5,5))

#Anonymus Function
liste=[2,4,6]
x=map(lambda i:i+5,liste)
print(list(x))   

#zip
liste2=[1,3,5]
z=zip(liste,liste2)
z_list=list(z)
print(z_list) 
#unzip
unZip=zip(*z_list)
unListe,unListe2=list(unZip)
print(unListe)
print(unListe2)
print(type(unListe2))
print(type(list(unListe)))

data.columns
dataFrame=data['Happiness.Score']
print(type(dataFrame))
#print(dataFrame)
data['happiness']=['Good'if i>5 else 'Bad' for i in dataFrame]
data.loc[:,['happiness','Happiness.Score']]
data.describe()

data.shape
data.tail()
data.boxplot(column='Trust..Government.Corruption.',by='happiness')
#melt
newData=data.head(10)
melted=pd.melt(frame=newData,id_vars='Country',value_vars=['Generosity','Happiness.Score'])
melted
melted.pivot(index='Country',columns='variable',values='value')
#CONCATENATING DATA
data.columns
data1=data['Economy..GDP.per.Capita.'].head()
data2=data['Health..Life.Expectancy.'].head()
concData=pd.concat([data1,data2],axis=1)
concData
#data['Health..Life.Expectancy.'].value_counts(dropna=False)
data['Health..Life.Expectancy.'].dropna(inplace=True)
assert data['Health..Life.Expectancy.'].notnull().all
data.columns
data1=data.loc[:,["Happiness.Score","Freedom","Economy..GDP.per.Capita."]]
data1.plot()
data1.plot(subplots=True)
plt.show()
data1.plot(kind="scatter",x="Economy..GDP.per.Capita.",y="Freedom")
plt.show()
fig, axes=plt.subplots(nrows=2,ncols=1)
data1.plot(kind="hist",y="Freedom",normed=True,bins=75,ax=axes[0])
data1.plot(kind="hist",y="Freedom",normed=True,bins=75,ax=axes[1],cumulative=True)
plt.savefig('graph.png')
plt
data.loc[5,["Freedom"]]
data.loc[1:10,"Happiness.Score":"Freedom"]
data.loc[1:10:-1,"Happiness.Score":"Freedom"]
data.loc[1:10,"Happiness.Score"]






