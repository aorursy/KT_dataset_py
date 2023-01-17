# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data.csv')

data.info()
#we have to 18207 entries and 89 colums

data.corr()
data.head(10)
data.columns
data.plot(kind='scatter',x='Balance',y='Finishing',alpha=0.5,color='red')

plt.xlabel('Positioning')

plt.ylabel('Finishing')

plt.title('Positioning & Finishing Scatter Plot')
print(data.Nationality.unique())
Turkey=data[data.Nationality == 'Turkey']
print(Turkey)
plt.plot(Turkey.GKHandling,Turkey.GKReflexes,color='red',label='Turkey')

plt.xlabel('GKHandling')

plt.ylabel('GKReflexes')

plt.show()
dictionary={'Club':'RealMadrid','Nationality':'Brazil'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Club']='Barcelona' #We're changing value RealMadrid to Barcelona

print(dictionary.values())
dictionary['Position']='ST' #Add new key and value

print(dictionary.values())
del dictionary['Position'] #delete Position key and value

print(dictionary)
dictionary.clear() #clear dictionary keys and values

print(dictionary)
print(dictionary)
series=data['Value'] #create series

print(type(series))
data_frame=data[['Value']] #create data frame

print(type(data_frame))
x=data['Finishing']>90 #Filtering

print(x)

data[x]