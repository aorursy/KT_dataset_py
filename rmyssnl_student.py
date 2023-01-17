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
# Read the data from the csv file  

data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
# First 5 rows

# data.iloc[0:3] Can also be written as

data.head()
# Last 5 rows

data.tail()
# Random rows

data.sample(3)
# Learn about the size of the .csv file

data.shape
# Datacontrol

# Look for missing values

data.isnull().values.any()
# List for missing values

data.isnull().sum()
# To get a short summary of the dataframe

data.info()
# datatype of the dataframes

data.dtypes
# datatype of the dataframes

data.iloc[:,0:5].dtypes
# Learning the datatype with iloc

data.iloc[:,5:8].dtypes
# Used to view some basic statistical details

data.describe()
# Used to find the pairwise correlation of all columns in the dataframe

data.corr()
#correlation map

sns.heatmap(data.corr(), annot=True, fmt= '.1f')

plt.show()
# Learning columns

data.columns
# Merge with columns _

data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]
# Convert strings in the Series/Index to be capitalized.

data.columns = [each.title() for each in data.columns]
# Show columns

for i,col in enumerate(data.columns):

    print(i+1,". columns ",col)
# Show unique Gender

data['Gender'].unique()
# Show count Gender

data['Gender'].value_counts()
# Show count Lunch

data['Lunch'].value_counts()
# Show count Race/Ethnicity

data['Race/Ethnicity'].value_counts()
b=(data['Writing_Score']<50).value_counts()

b
x=data['Race/Ethnicity'].value_counts().values

x
data.Math_Score.plot(kind = 'line', color = 'lime',label = 'Math_Score',linewidth=1,alpha = 1,grid = True,linestyle = ':')

#data.Reading_Score.plot(color = 'c',label = 'Reading_Score',linewidth=1, alpha = 1,grid = True,linestyle = '-.')

data.Writing_Score.plot(color = 'blue',label = 'Writing_Score',linewidth=1, alpha = 0.5,grid = True,linestyle = '--')

plt.legend(loc='lower center')

plt.xlabel('rows')

plt.ylabel('score')

plt.title('Math_Score & Writing_Score')

plt.show()
data.plot(kind='scatter',x='Reading_Score',y='Writing_Score', alpha=.7, color='magenta', label = 'kasjsk')

# plt.xlabel("Reading_Score") not writting because x belive 

plt.title('scatter plot')

plt.legend(loc='right')

plt.show()
data.Math_Score.plot(kind='hist', bins=50, facecolor = "blue", alpha=.5, grid = True)

plt.show()