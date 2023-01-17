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

from subprocess import check_output

plt.show()
data = pd.read_csv('../input/creditcard.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, linewidths=.3,fmt='.1f' , ax=ax)

plt.show()
data.head()
data.columns
#data.V4.plot(kind='line', color= 'red', label='V4', linewidth=1, alpha=1, grid=True, linestyle='-')

data.V5.plot(kind='line', color='blue', label='V5', linewidth=1, alpha=1, grid=True, linestyle='--')

#data.V6.plot(kind='line', color='green',label='V6', linewidth=1, alpha=0.7, grid=True, linestyle=':')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot of Credit Carts Fraud Detections')

plt.show()
#data.V4.plot(kind='line', color= 'red', label='V4', linewidth=1, alpha=1, grid=True, linestyle='-')

#data.V5.plot(kind='line', color='blue', label='V5', linewidth=1, alpha=1, grid=True, linestyle='--')

data.V6.plot(kind='line', color='green',label='V6', linewidth=1, alpha=0.7, grid=True, linestyle=':')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot of Credit Carts Fraud Detections')

plt.show()
data.Amount.plot(kind='line', color= 'red', label='Amount', linewidth=1, alpha=1, grid=True, linestyle='-')

#data.V5.plot(kind='line', color='blue', label='V5', linewidth=1, alpha=1, grid=True, linestyle='--')

#data.V6.plot(kind='line', color='green',label='V6', linewidth=1, alpha=0.7, grid=True, linestyle=':')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot of Credit Carts Fraud Detections')

plt.show()
data.plot(kind='scatter', x='Amount', y='V6', alpha=0.6, color='orange')

plt.xlabel('Amount')

plt.ylabel('V6')

plt.title('Amount-V6 Scatter Plot')

plt.show()
data.Amount.plot(kind='hist',bins=25, figsize=(10,10))

plt.show()
data.V8.plot(kind='hist', bins=25)

plt.clf()
data[np.logical_and(data['Amount']>10000, data['V2']<0)]
data[(data['Amount']>25000) & (data['V17']<1)]
lis = ['cars','planes','trucks','ships','quadcopters']

for vehicles in lis:

    print('vehicles is: ',vehicles)

print("This from vehicle list...")

print("...........................................")

num = 50

while num != 110 :

    print('num is: ',num)

    num +=5 

print('last_index is:',num)

print("...........................................")

for index,value in data[['Amount']][0:3].iterrows():

    print(index," : ",value)
dictionary={'monday':'day1', 'tuesday':'day2','wedn':'day3'}

print(dictionary.keys())

print(dictionary.values())

print("...........................................")

dictionary['monday']="first_dayOfweek"           #change value

for key,value in dictionary.items():

    print(key,":",value)

print("...........................................")

dictionary['thursday']="day4"                    #add

for key,value in dictionary.items():

    print(key,":",value)

print("...........................................")

del dictionary['tuesday']                        #del

for key,value in dictionary.items():

    print(key,":",value)

print("...........................................")

print('monday' in dictionary)                   #check

#last

dictionary.clear()

print("...........................................")

print(dictionary)