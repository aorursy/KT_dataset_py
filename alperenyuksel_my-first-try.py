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
import pandas as pd

data = pd.read_csv("../input/fifa19/data.csv")

data
data.info()
data.columns
data.head(10)

#First 10 players in this dataset
data.tail(10)

#Last 10 players in this dataset
len(data)
type(data)
max(data)
min(data)
data.shape
data[["Name"]]

#Only Name column
data[["Name","Overall"]]

#Only Name and Overall columns that we want to see
a=data['Overall']>85

data[a]

#Players who have overall more than 85
data[np.logical_and (data['Overall']>85,  data['Age']<23)]

#Players who have overall more than 85 and younger than 23.
for index, value in data[['Name']][97:98].iterrows():

    print(index, ":", value)#If we want to see the player who is in 97th index.
data.corr()

#Correlation info
f, ax=plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, linewidth=7, fmt='.1f', ax=ax)

plt.show()

#We can make some inferences about there is a correlation between shot power and long shots, positioning and finishing.
data.Stamina.plot(kind='line', color='g', label='Stamina', linewidth=1, alpha=0.5, grid=True, linestyle=':')

data.Strength.plot(color='r', label='Strength', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')             

plt.ylabel('y axis')

plt.title('line plot')

plt.show()#a line plot for stamina and strength
data.plot(kind='scatter', x='Stamina', y='Strength', color='black', alpha=0.2, grid=True)

plt.xlabel('Stamina')

plt.ylabel('Strength')

plt.title='Stamina and Strength'



plt.show()

#Relationship between Stamina and Strength columns
data.plot(kind='scatter', x='Age', y='SprintSpeed', color='blue', alpha=0.1)

plt.xlabel('Age')

plt.ylabel('SprintSpeed')

plt.title="Age and SprintSpeed"



plt.show()

#Relationship between Age and SprintSpeed columns
data.Overall.plot(kind='hist', bins=70, figsize=(15,15))

#Overalls and sum of players (Most players have 66th or 67th overall)
data.Overall.plot(kind='hist', bins=70, figsize=(15,15))

plt.clf()

#clear