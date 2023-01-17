# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart.csv')
data.info()
data.head(10)

data.corr()
data.columns

# Line Plot



data.age.plot(kind = 'line', color = 'b' , label = 'Age',linewidth = 1, linestyle=':',grid= True)

data.trestbps.plot(label='Trestbps', color = 'r', linewidth = 1,linestyle='-',grid=True)

plt.xlabel ('x axis')

plt.ylabel('y axis')

plt.title('Age Trestbps Line Plot')

plt.show()

#Scatter Pilot



data.plot(kind='scatter',x = 'thalach', y = 'trestbps', color = 'g',alpha = 0.4)

plt.xlabel('thalach')

plt.ylabel('trestbps')

plt.title('Thalach Trestbps Scatter Pilot')

plt.show()
#Histogram

data.chol.plot(kind = 'hist', bins = 50 , figsize = (15,15),color = 'c')

data.thalach.plot(kind = 'hist', bins = 50 , figsize = (15,15),color = 'r',alpha=0.5)

plt.show()
#clf = cleans it up again

data.chol.plot(kind = 'hist', bins = 50 , figsize = (15,15),color = 'c')

data.thalach.plot(kind = 'hist', bins = 50 , figsize = (15,15),color = 'r',alpha=0.5)

plt.clf()
#create new dictionary 

dictionary = {'heart diseases':'cholesterol' , 'brain diseases' : 'cerebral hemorrhage'}

print(dictionary.keys())

print(dictionary.values())
dictionary['heart diseases'] = "heart attack"

print(dictionary)

dictionary['stomach diseases'] = "gastrospasm"

print(dictionary)

del dictionary ['brain diseases']

print(dictionary)

print('brain diseases' in dictionary)

dictionary.clear()

print(dictionary)
#del dictionary

print (dictionary)
data=pd.read_csv('../input/heart.csv')
x = data['age'] >= 50

data[x]
y = data['chol'] <=200

data[y]

print(data[y])
data[(data['age'] >=60) & (data['chol']<=200)]
data[(data['sex'] == 1) & (data['thalach'] > 180 ) & (data['chol'] > 250)]

print(data[(data['sex'] == 1) & (data['thalach'] > 180 ) & (data['chol'] > 250)])

for index,value in data[['trestbps']][0:5].iterrows():

    print(index," : ",value)
for index,value in data[['chol']][0:2].iterrows():

    print(index," : ",value)