# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')

data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(25, 25))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(5)




data.columns

data.SprintSpeed.plot(kind='line', color='g', label='SprintSpeed', linewidth=1, alpha=0.5, grid=True , linestyle=':')

data.Balance.plot(kind='line', color='r', label='Balance', linewidth=1, alpha=0.5, grid=True , linestyle='-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# Scatter Plot 

# x = SprintSpeed, y = Balance

data.plot(kind='scatter', x='SprintSpeed', y='Balance', alpha=0.5, color='red')

plt.xlabel('SprintSpeed')

plt.ylabel('Balance')

plt.title('SprintSpeed Balance Scatter Plot')

plt.show()

#Histogram

#bins = number of bar in figure

data.SprintSpeed.plot(kind='hist',bins = 50, figsize =(20,12))

plt.xlabel('SprintSpeed')

plt.show()
data.columns
# 1 - Filtering Pandas data frame

x = data['SprintSpeed'] > 85

y = data['Finishing'] > 85

z = data['ShotPower'] > 85

data[x & y & z]
result = data[x & y & z]

result.columns

result['SprintSpeed']

myList = result['SprintSpeed']





print('SprintSpeed with Condition')

print('')

for mySpeed in myList:

    if mySpeed > 93:

        print('SprintSpeed : ',mySpeed)

    else:

        print('SprintSpeed is not greater then 93, the value is :', mySpeed)

        

print('')       

print('SprintSpeed with Index')

print('-----------------------')

for index ,value in enumerate(result['SprintSpeed']):

    print('Index: ', index, 'SprintSpeed: ', value)

        
