# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualizing processes


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/data.csv") # you can add data with using pandas library easily

# Any results you write to the current directory are saved as output.
data.info()
data.head(5)
data.columns
data.corr()
data.texture_se.plot(kind='line', color='green', label='texture_se', linewidth=2, alpha=0.5, grid=False, linestyle='-')
data.texture_se.plot(kind='line', color='green', label='texture_se', linewidth=2, alpha=0.5, grid=False, linestyle='-')
data.smoothness_se.plot(kind='line', color='red', label='smoothness_se', linewidth=2, alpha=0.5, grid=False, linestyle=':')
plt.xlabel('x')              # label = name of label
plt.ylabel('y')
plt.title('Line Plot with Texture and Smoothness')            # title = title of plot
plt.show()
data.perimeter_mean.plot(kind='hist', bins=10, figsize=(10,10))
data.plot(kind='Scatter', x='area_mean', y='symmetry_worst', alpha=0.7, color='blue', grid=True)
plt.scatter(data.area_mean, data.symmetry_worst)
dictionary = {'type': 'malignant', 'area': 'breast'}
print(dictionary)
print("\n")
print(dictionary.keys()) #prints keys
print(dictionary.values()) #prints values

dictionary['type'] = 'benign' #update dictionary
print("\n")
print(dictionary.keys())
print(dictionary.values())

del dictionary['area'] #deletes area's key & value
print("\n")
print(dictionary.keys())
print(dictionary.values())
malignant = data['diagnosis'] =='M'
data[malignant]



data[(data['diagnosis']=='B') & (data['radius_mean']>15)]

i = 0
while i != 5 :
    print(data['id'][i]) #print id's for first 5 rows
    i +=1 
dictionary = {'type': 'malignant', 'area': 'breast', 'tumor_radius': '14'}
for key,value in dictionary.items():
    print(key," : ",value)