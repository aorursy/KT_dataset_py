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



data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.info()
data.corr()
#correlation map



f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(7)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.restecg.plot(kind = 'line', color = 'g',label = 'restecg',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.oldpeak.plot(color = 'r',label = 'oldpeak',linewidth=1, alpha = 0.2,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='trestbps', y='chol',alpha = 0.5,color = 'red')

plt.xlabel('trestbps')              # label = name of label

plt.ylabel('chol')

plt.title('trestbps chol Scatter Plot')            # title = title of plot
# clf() = cleans it up again you can start a fresh

data.trestbps.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'turkey' : 'trabzon','georgia' : 'batumi'}

print(dictionary.keys())

print(dictionary.values())
# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
series = data['exang']        # data['Defense'] = series

print(type(series))

data_frame = data[['exang']]  # data[['Defense']] = data frame

print(type(data_frame))

# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data['trestbps']>180     # There are only 2 patients who have higher defense value than 180

data[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['trestbps']>165) & (data['chol']>150)]



#filtering with pandas

#data[np.logical_and(data['Defense']>200, data['Attack']>100 )]   
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')