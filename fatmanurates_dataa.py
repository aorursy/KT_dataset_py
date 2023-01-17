# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.info()
data.head()
data.head(3)

data.tail()
data.tail(3)
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)
plt.show()
data.columns
data.diagnosis
data.diagnosis.unique()

data.describe()
#Line plot
data.symmetry_worst.plot(kind = 'line', color = 'black',label = 'symmetry_worst',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')
data.smoothness_mean.plot(kind = 'line', color = 'orange', label = 'smoothness_mean', linewidth=1, alpha = 0.7, grid = True)
data.fractal_dimension_se.plot(kind = 'line', color = 'blue', label = 'fractal_dimension_se', linewidth=1, alpha=0.7, grid = True, linestyle = '-.')

plt.legend()
plt.xlabel("x_axis")
plt.ylabel("y_axis")
plt.title("Line_Plot")
plt.show()

#Scatter Plot
plt.scatter(data.symmetry_worst, data.smoothness_mean, color="red",label ="data")
plt.xlabel('x_axis')
plt.ylabel('y_axis')
plt.show()
#Scatter plot
data.plot(kind='scatter', x='symmetry_worst', y='smoothness_mean', alpha=0.5, color='green')
plt.xlabel('x_axis')
plt.ylabel('y_axis')
plt.show()
# histogram
data.perimeter_worst.plot(kind = 'hist',bins=150, figsize=(20,20))
plt.show()
dictionary = {'black' : 'siyah','green' : 'yeşil','pink' : 'pembe', 'red' : 'kırmızı','blue' : 'mavi'}
print(dictionary.keys())
print(dictionary.values())
#update existing entry
dictionary['black'] = 'En koyu renk'
print(dictionary)
#add new entry
dictionary['purple'] = 'mor'
print(dictionary)
# remove entry
del dictionary['green']
print(dictionary)
print('purple' in dictionary)
#remove all entries dictionary
dictionary.clear()
print(dictionary)



def animals():
    dic = {'cat' : 'kedi', 'dog':'köpek','horse':'at','sheep':'koyun'}
    return dic
print(animals())
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.columns
series = data['Pregnancies']
print(series)
print(type(series))
data_frame = data[['Pregnancies']]
print(data_frame)
print(type(data_frame))
print(7 == 8)
print(8 != 15)
print(True and False)
print(False or True)
x = data['Pregnancies']>13
data[x]
data[np.logical_and(data['Pregnancies']>13, data['Glucose']<160)]
data[(data['Pregnancies']>13) & (data['Glucose']<160)]
i = 0
while i < 100:
    i = i*5
    print(i)
    i +=1 
print(i)
name = ['irem','aybüke','ebrar','vuslat','şevval']
for i in name:
    print(i)
 
for index,value in enumerate(name):
    print(index,":",value)

dictionary = {'black' : 'siyah','green' : 'yeşil','pink' : 'pembe', 'red' : 'kırmızı','blue' : 'mavi'}
for key,value in dictionary.items():
    print(key,':',value)

