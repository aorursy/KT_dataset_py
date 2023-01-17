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
data = pd.read_csv('/kaggle/input/fifa19/data.csv') # Read a comma-separated values (csv) file into DataFrame.

data.head(10) # This function returns the first 10 rows for the object based on position.
data.info()
data.corr() #Compute pairwise correlation of column.
#correlation map

f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(), annot =True, linewidths = .5, fmt = '.1f', ax=ax) #Plot rectangular data as a color-encoded matrix.

plt.show()
data.columns
data.Finishing.plot(kind = 'line', label = 'Finishing', color= 'g', linewidth=1, alpha = 0.5, grid=True, linestyle=':',figsize=(15,15))

data.Acceleration.plot(label = 'Acceleration', color= 'red', linewidth=1, alpha = 0.5, grid=True, linestyle='-')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#Scatter Plot

#x = attac, y = defense

data.plot(kind='scatter',x='Age', y='Overall',alpha=0.7,color='red',figsize=(15,15))

plt.xlabel('Finishing')

plt.ylabel("Overall")

plt.title('Age - Overall Scatter Plot')

plt.show()
# Histogram

# bins = number of bar in figure

data.Jumping.plot(kind='hist', bins = 100, figsize=(15,15))

plt.show()
data.Jumping.plot(kind = 'hist', bins = 50)

plt.clf()
data.head()
dictionary = {'L. Messi':'31','Cristiano Ronaldo':'33','Neymar Jr':'260','De Gea':'27'}

print(dictionary.keys())

print(dictionary.values())

dictionary['K. De Bruyne'] = '27' #Add

print(dictionary)

dictionary['L. Messi'] = '32'#Change

print(dictionary)

print('Cristiano' in dictionary)

dictionary.clear()

print(dictionary)
series = data['Age']

print(type(series))

data_frame = data[['Age']]

print(type(data_frame))
x = data['Age']

data[x > 43]
data[(data['Age'] < 20) & (data['Overall'] > 80)]