# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv", delimiter=",")
data.info() 
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(30,30))

sns.heatmap(data.corr(), annot=True, linewidths=5, fmt='.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Age.plot(kind = 'line', color = 'g',label = 'Age',linewidth=2,alpha = 0.5,grid = True,linestyle = ':')

data.Overall.plot(color = 'r',label = 'Overall',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('age')              # label = name of label

plt.ylabel('Overall')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot

# x = Volleys, y = Dribbling

data.plot(kind='scatter', x='Volleys', y='Dribbling',alpha = 0.5, color = 'blue')

plt.xlabel('Volleys')

plt.ylabel('Dribbling')

plt.title('Volleys Dribbling Scatter Plot')

plt.show()
# Histogram

# bins = number of bar in figure

data.ShotPower.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
#create dictionary and look its keys and values

dictionary = {'Name' : 'Age','Nationality' : 'Flag'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['Name'] = "Age"    # update existing entry

print(dictionary)

dictionary['Nationality'] = "Flag"       # Add new entry

print(dictionary)
# 1 - Filtering Pandas data frame



x = data['Age']>30    

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 FIFA19 who have higher Age value than 30 and higher Potential value than 70

data[np.logical_and(data['Age']<30, data['Potential']>70 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Age']>30) & (data['Potential']>80)]
# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'Name':'Age','Nationality':'Flag'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['Wage']][0:3].iterrows():

    print(index," : ",value)