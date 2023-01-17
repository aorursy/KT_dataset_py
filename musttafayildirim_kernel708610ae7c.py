# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
#Line Plot

#color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.sepal_length.plot(kind = 'line', color = 'g',label = 'sepal length',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.petal_length.plot(color = 'r',label = 'petal length',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.sepal_length.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
#create dictionary and look its keys and values

dictionary = {'sepal_length' : '5.1','sepal_width' : '3.5'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['ulke'] = "Türkiye"    # update existing entry

print(dictionary)

dictionary['il'] = "İstanbul"       # Add new entry

print(dictionary)

del dictionary['ulke']              # remove entry with key 'spain'

print(dictionary)

print('il' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
x = data['sepal_length']==5

data[x]
data[np.logical_and(data['sepal_length']==7.9, data['petal_length']==6.4 )]
data[(data['sepal_length']==7.9) & (data['petal_length']==6.4 )]
# Example of list comprehension

num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)

# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
print(len(petal_width))
data.tail()
data.shape
data.info()
data.describe()
data.boxplot(column = "sepal_length",by = "sepal_width")
data_new = data.head()

data_new
melted = pd.melt(frame=data_new,id_vars = 'species', value_vars= ['sepal_length','sepal_width'])

melted
data.info()
data1 = data.loc[:,['sepal_length','sepal_width','petal_length','petal_width']]

data1.plot()
data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x="sepal_length",y = "sepal_width")

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "sepal_length",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "sepal_length",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()