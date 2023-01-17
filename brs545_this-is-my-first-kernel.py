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
data=pd.read_csv('../input/data.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(25,25))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns

data.Overall.plot(kind = 'line', color = 'g',label = 'Overall',linewidth=3,alpha = 0.5,grid = True,linestyle = ':')

data.Special.plot(color = 'r',label = 'Special',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='Overall', y='Special',alpha = 0.5,color = 'red')

plt.xlabel('Overall')              # label = name of label

plt.ylabel('Special')

plt.title('Overall Special Scatter Plot')
plt.scatter(data.Overall,data.Special,color='blue',alpha=0.3)

plt.show()
data.Overall.plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.show()
data.Overall.plot(kind = 'hist',bins = 50)

plt.clf()
dictionary={'Value':127.1, 'Overall':94,'Dribbling':88,'Finishing':94}

print(dictionary.keys())

print(dictionary.values())
dictionary['Value']=110.5

print(dictionary)

dictionary['Potential']=93

print(dictionary)

del dictionary['Potential']

print(dictionary)

print('Value' in dictionary)

print('value' in dictionary)

dictionary.clear()

print(dictionary)

print(dictionary)
data = pd.read_csv('../input/data.csv')
series = data['Overall']       

print(type(series))

data_frame = data[['Special']]  

print(type(data_frame))
print(9 < 3)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
x = data['Overall']>90     # There are only 3 pokemons who have higher defense value than 200

data[x]
data[np.logical_and(data['Overall']>90, data['Finishing']>86)]
data[(data['Overall']>90) & (data['Finishing']>86)]
i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
listt = [1,2,3,4,5,6,7,8,9]

for i in listt:

    print('i is: ',i)

print('')
for index,value in enumerate(listt):

    print(index, ' : ', value)
for index,value in data[['Overall']][0:1].iterrows():

    print(index," : ",value)