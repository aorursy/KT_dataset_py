# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  #

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#we use pandas for csv data 

#pandas include read_csv that is built in scope for pyhton

data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
# we can see data's infos :data.info()

data.info()

# we can see data from creditcard.csv  :data.head()

data.head()#default we can see five rows

#or 

data.head(10)#we can see ten rows
#we can see data's relations

data.corr()

#this csv has not got losts of corrolations
#corralation map

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
#we can see creditcard.csv's columns title

data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Amount.plot(kind = 'line', color = 'r',label = 'Amount',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.V4.plot(color = 'r',label = 'V4',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper left')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Amount, y = V4

data.plot(kind='scatter', x='Amount', y='V4',alpha = 0.5,color = 'red')

plt.xlabel('Amount')              # label = name of label

plt.ylabel('V4')

plt.title('Amount V4 Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.V4.plot(kind = 'hist',bins = 30,figsize = (12,12))

plt.show()
dictionary={'Mercedes':'S400','BMW':'740i','Ferrari':'F30'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Mercedes']='S500'  #update existing entry

print(dictionary)

dictionary['Opel']='Astra'    #add new entry

print(dictionary)

del(dictionary['BMW'])        #remobe entry with key 'BMW'

print(dictionary)

print('Opel' in dictionary)  # check entry

dictionary.clear()           #clean dictionary

print(dictionary)
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

data.head(10)
data_series=data['Amount']  #series 

print(type(data_series))

data_frime=data[['Amount']]

print(type(data_frime))
#Filtering Pandas data frame

amount=data['Amount']>20000

data[amount]
# 2 - Filtering pandas with logical_and

data[np.logical_and(data['Amount']>10000,data['V4']>10)]
#  '&' is operater that like 'np.logical_and'

data[(data['Amount']>10000) & (data['V4']>10)]
#while 

i=0

x=data[(data['Amount']>10000) & (data['V4']>10)]

print(len(x))

while i< len(x):

    print(x['Amount'])# why not run this code ?  ---print(x['Amount'][i])

    i+=1
#For



# For pandas we can achieve index and value

for index,value in data[['Amount']][0:2].iterrows():

    print(index," : ",value)

print()



x=data[(data['Amount']>10000) & (data['V4']>10)]

for i in x.index:

    print(x['Amount'][i])

print()    

    

lis ={1,2,3,4,5}

for i in lis:

    print(i)

print()

# Enumerate index and value of list 

for index,value in enumerate(lis):

    print(index ,": ", value)

print()



dictionary ={'Mercedes':'CLK','BMW':'740'}

for key,value in dictionary.items():

    print (key,": ", value)

print()
