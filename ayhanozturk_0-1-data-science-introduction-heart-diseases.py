import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data =pd.read_csv("../input/heart.csv")
print(data.info())

print(data.describe())
data.shape
data.corr()
data.columns


a,b = plt.subplots(figsize=(10,10))# figsize meaning is determine size of shape

sns.heatmap(data.corr(),annot=True,linewidths=0.9,fmt='.3f',ax=b) 

plt.show()
data.head(100).loc[20:25,["slope","oldpeak","target","cp"]]

new = pd.concat([data.tail(5).loc[::-1,["slope","oldpeak","target","cp"]],data.head(5).loc[::-1,["slope","oldpeak","target","cp"]]],axis = 0)

print(new)

new["loop"]=["tr_" if ((oldpeak >0.8) & (slope == 1)) else "fal_" for slope,oldpeak,target,cp in new.values]

   
print(new)
new.drop(["loop"],axis=1,inplace = True)
print(new)
data.slope.plot(kind='line',color='r',label='slope',grid=True,alpha=1,figsize=(5,5),linewidth=0.8,linestyle=':')

plt.legend(loc='lower left')

plt.xlabel('xlabel')

plt.ylabel('ylabel')

plt.title('title')

plt.show()
data.plot(kind='scatter',color='r',label='scatter',grid=True,alpha=0.9,figsize=(5,5),x='target',y='cp')

plt.legend(loc='upper center')

plt.xlabel('xlabel')

plt.ylabel('ylabel')

plt.title('title')

plt.show()
data.plot(kind='line',color='r',label='scatter',grid=True,alpha=0.9,x='target',y='cp',linewidth=0.9,linestyle=':')

plt.legend(loc='upper center')

plt.xlabel('xlabel')

plt.ylabel('ylabel')

plt.title('title')

plt.show()
data.target.plot(kind='hist',color='r',label='hist',grid=True,alpha=1,figsize=(5,5),bins=10)
#Dictionary



dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain']='london'

print(dictionary)
dictionary['france']="paris"

print(dictionary)
del dictionary['spain']

print(dictionary)
print('france' in dictionary) 

print(dictionary)
dictionary.clear()                   # remove all entries in dict

print(dictionary)
variable_1=data['target']

print(type(variable_1))

variable_2=data[['target']]

print(type(variable_2))
print(variable_2)
x=variable_2['target']>0

print(x)
print(variable_2[x])
data[(data['target']>0) & (data['cp']>2)]
print(type(data))
i=0

while i!=5:

    print('i is = ',i)

    i+=1

print('i is equal', i)

    
lis = [1,2,3,4,5]

for y in lis:

    print('i is: ',i)

print('')



dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



for index,value in data.loc[0:3,['cp']].iterrows(): 

    print('index = ',index,' value = ',value)
data.dtypes