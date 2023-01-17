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
data = pd.read_csv("../input/diabetes.csv") 
data.info() # gives info about our data
data.corr()
#correlation map 
f,ax = plt.subplots(figsize=(15, 10)) #figsize; sets the size of boxes
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#annot = True; allows the use of correlation results in boxes
#linewidth; thickness of line between boxes
#fmt; sets the length of the decimal portion
plt.show()
print(data.head(20)) #Gives us top 20 of data.
print(data.tail(20)) #gives us last 20 of data.
data.Age.plot(kind = 'hist',bins = 60,figsize = (15,15))
# bins = number of bar in figure
#x axis is Age. y axis is frequency of Age.
plt.show()
data.Outcome.plot(kind = 'hist',bins = 20,figsize = (5,5))
# bins = number of bar in figure
#x axis is Age. y axis is frequency of Age.
plt.show()
# x = pregnancy, y = outcome
data.plot(kind='scatter', x='Glucose', y='Insulin',alpha = 0.5,color = 'red')
#plt.scatter(data.Glucose,data.Insulin,alpha = 0.5,color = 'red') ## It is same as top row
plt.xlabel('Glucose')              
plt.ylabel('Insulin')
plt.title('Glucose and Insulin Relevant')            
plt.show()
data.describe()
filt = data.Age > data.Age.mean() #if da
filtered_data = data[filt]
print(filtered_data)
x = data['Glucose']>185 
data[x]
data[np.logical_and(data['Glucose']>180, data['Outcome'] == 1 )] 
#we have find the people who has glucose level over 180 and have diabetes.

data[np.logical_and(data['Age']>40, data['Outcome'] == 0 )]
#we have find the people who is over 40 years old and have not diabetes.