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
data = pd.read_csv('../input/insurance.csv')
data.info
data.corr()

# correlation map
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
data.describe()
data.head(16)
#line plot

data.age.plot(kind= 'line', color = 'g', label = 'age', linewidth=1, alpha=0.5, grid= True , linestyle='-.' )
data.bmi.plot(color = 'r', label = 'bmi', linewidth=1, alpha=0.5, grid= True, linestyle=':' )
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('age of bmi')
plt.show()
# scatter plot
# x=age y=charges

data.plot(kind='scatter', x = 'age', y='charges', alpha=0.5, color='b')
plt.xlabel('age')
plt.ylabel('charges')
plt.title('medical cost by age')

# histogram
data.age.plot(kind='hist', bins=50, figsize=(10,10))
plt.show()
data.age.plot(kind = 'hist',bins = 50)
plt.clf()


mean_age = np.mean(data.age)
print("mean_age=",mean_age)
mean_medical_charges=data.charges.mean()
print("mean_medical_charges=",mean_medical_charges)
# creta dictionry
dictionary = {'region':'southwest','northwest':'southeast'}
print(dictionary.keys())
print(dictionary.values())
dictionary['region'] = "izmir"
print(dictionary)
dictionary['akdeniz'] = "antalya"
print(dictionary)
del dictionary['northwest']
print(dictionary)
print('region' in dictionary)
print('antalya' in dictionary) # why value doesn't see
print('15' in dictionary)
dictionary.clear()
print(dictionary)
data = pd.read_csv('../input/insurance.csv')
data.describe()
(data['age']/2).describe()
series = data['bmi']
type(series)
series1 = data['region']
print(type(series1))
data_frame = data[['region']]
print(type(data_frame))
print(5>3)
print(4<=4)
print(3!=2)
print(True and False)
print(True or False)
x = data['age']<mean_age
data[x]
data[np.logical_and(data['age']<mean_age,data['bmi']<30)]
# How to select 3 parameters?
data[np.logical_or(data['age']<mean_age,data['bmi']<30)]

data[(data['age']<20) & (data['smoker']=="yes")]
i=0
while i !=5:
    print('i is:',i)
    i +=1
print(i,'is equal to 5')
list_age = list(data['age'])
age_21=21
i=0
for age_21 in list_age:
    if (age_21==21):
        i=i+1

print('age_21:',age_21)
# smoker and no smoker
d=0
a=0
for each in data.smoker:
    if (each=="yes"):
        d=d+1
    else:
        a=a+1
        continue
print("smoker:",d)
print("no_smoker:",a)

# children and no children
c=0
b=0
for each in data.children:
    if (each==0):
        b=b+1
    else:
        c=c+1
        continue
print("children:",c)
print("no_children:",b)
#for index,value in enumerate(list_age):
 #   print(index,":",value)
#print()
print(dictionary)
dictionary = {'spain':'madrid','turkey':'antalya','ege':'izmir'}
for key,value in dictionary.items():
    print(key,":",value)
print()
for index,value in data[['bmi']][0:3].iterrows():
    print(index,":",value)
for index,value in data[['sex']][5:7].iterrows():
    print(index,":",value)
data['children'].value_counts()
# Number of each type of column
data.dtypes.value_counts()
# Number of unique classes in each object column
data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
data.groupby(['region'])[['bmi']].agg(['mean','median','count'])