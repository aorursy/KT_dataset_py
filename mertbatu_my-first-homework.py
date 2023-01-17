# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for basic visualization

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.info()
data.head()
data.corr()
#Correlation Map



f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot= True, linewidths= .5, fmt='.1f', ax= ax)

plt.show()
data.head(10)
data.columns
data.Age.plot( kind='line', color='blue', label='Age', linewidth=1, alpha=.5, grid= True, linestyle=':')

data.TotalWorkingYears.plot(kind='line', color= 'green', label= 'Total Working Years', linewidth=1, alpha=.5, grid= True, linestyle='-.')



plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('Number of Employees')              # label = name of label

plt.ylabel('Years')

plt.title('Line Plot of the Years-Employees Relation')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Age, y = Total Working Years

data.plot(kind='scatter', x='Age', y='TotalWorkingYears',alpha = 0.5,color = 'red')

plt.xlabel('Age')              # label = name of label

plt.ylabel('Total Working Years')

plt.title('Age-Total Working Years Scatter Plot')            # title = title of plot

plt.show()
data.Age.plot(kind='hist', color = 'MediumSeaGreen', bins=60, figsize = (20,12))

plt.show()
data.Age.plot(kind = 'hist',bins = 50)

plt.clf()
series = data['Age']        # data['Defense'] = series

print(type(series))

data_frame = data[['Age']]  # data[['Defense']] = data frame

print(type(data_frame))
x = data['Age'] > 59

data[x]
data[np.logical_and(data['Age']>59, data['MaritalStatus'] == 'Married' )]
# First Type

data[np.logical_and(data['Age'] > 59, data['MaritalStatus'] == 'Married')]
# Second Type

data[(data['Age']> 59) & (data['MaritalStatus'] == 'Married')]
for index,value in data[['Age']][0:4].iterrows():

    print(index," : ",value)