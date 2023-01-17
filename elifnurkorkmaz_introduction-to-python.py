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
data=pd.read_csv('../input/world-happiness/2016.csv')
data.info()
#correlation map

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax,center=0) 

data.head()
#str.replace=change column names without brackest and space

data.columns = data.columns.str.strip().str.replace('(', '').str.replace(')', '').str.replace(' ', '')

data.head()
data.columns
#line plot 



#kind=kind of plot ,color=color of plot,label=label we draw,linewidth=width of line

#alpha=transparency,grid=is there grid(boolean),linestyle= shape of the line showing the label we have drawn

data.HappinessScore.plot(kind='line',color='r',label='HappinessScore',linewidth=1,alpha=0.5,grid=True,linestyle='-')

data.LowerConfidenceInterval.plot(kind='line',color='g',label='LowerConfidenceInterval',linewidth=1,alpha=0.5,grid=True,linestyle='-')

data.UpperConfidenceInterval.plot(kind='line',color='b',label='UpperConfidenceInterval',linewidth=1,alpha=0.5,grid=True,linestyle='-')

data.EconomyGDPperCapita.plot(kind='line',color='purple',label='EconomyGDPperCapita',linewidth=1,alpha=0.5,grid=True,linestyle='-')

data.Family.plot(kind='line',color='c',label='Family',linewidth=1,alpha=0.5,grid=True,linestyle='-')

data.HealthLifeExpectancy.plot(kind='line',color='m',label='HealthLifeExpectancy',linewidth=1,alpha=0.5,grid=True,linestyle='-')

data.Freedom.plot(kind='line',color='y',label='Freedom',linewidth=1,alpha=0.5,grid=True,linestyle='-')





plt.legend(loc='upper right') #legend=show label name into plot

plt.xlabel('x axis')          #label=name of label

plt.ylabel('y axis')

plt.title('Line Plot')        #title =title of plot

plt.figure(figsize=(10,20))
#scatter plot 



#1st plot

# x=Freedom,  y=HappinessScore

data.plot(kind='scatter',x='Freedom',y='HappinessScore',alpha=0.5,color='r')

plt.xlabel('Freedom')          #label=name of label

plt.ylabel('HappinessScore')

plt.title('Freedom-HappinessScore  Scatter Plot')        #title =title of plot



#2nd plot

# x=Freedom,  y=HappinessScore

data.plot(kind='scatter',x='EconomyGDPperCapita',y='HappinessScore',alpha=0.5,color='m')

plt.xlabel('EconomyGDPperCapita')          #label=name of label

plt.ylabel('HappinessScore')

plt.title('EconomyGDPperCapita-HappinessScore Scatter Plot')        #title =title of plot
#alternative scatter plot

plt.scatter(data.Freedom,data.HappinessScore,alpha=0.5,color='c')
#Histogram



#bins=number of bar in figure

#fizgsize =size of figure

data.HappinessScore.plot(kind='hist',bins=50 ,figsize=(7,7),color='m')
data[np.logical_and(data['HappinessScore']>7.501,data['Freedom']>0.57)]
#this is the same as above 'logical_and' control

data[(data['HappinessScore']>7.501) & (data['Freedom']>0.57)]
data[np.logical_or(data['HappinessScore']>7.501,data['Freedom']>0.57)]
#this is the same as  above 'logical_or' control

data[(data['HappinessScore']>7.501) | (data['Freedom']>0.57)]
#Stay in loop if condition(i is not equal 5) is true

i=0

while (i !=5):

    print ('i:',i)

    i+=1

print('i is equal 5')   
#Stay in loop if condition(i is not equal 5) is true

lis=[1,2,3,4,5]

for i in lis:

    print('i is:',i)

#Get index and value of List

#Enumerate index and value of list

#index : value = 0:1 , 1:2 , 2:3 , 3:4 , 4:5

for index,value in enumerate(lis): # get index of list and value of list

    print(index," : ", value)

#Get key and value of Dictionaries

#We can use for loop to achive key and value of dictionary.

#We learnt key and value at dictionary part.



dictionary={'spain':'madrid','france':'paris'}



for key,value in dictionary.items():

    print(key," : ",value)

    
#For pandas we can achieve index and value

for index,value in data[['HappinessScore']][0:5].iterrows():

    print(index," : ",value)