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
data = pd.read_csv('/kaggle/input/heart.csv')

data.info()
print(data)
data.corr()
describe = data.describe()

print(describe)
data.columns
filtre1 = data.sex != 0 # cinsiyeti erkek olanlar icin True ve False olarak verir.

filtrelenmis_data = data[filtre1] #cinsiyeti erkek olanlar icin True ve False larin degerini yani yasini gosterir.

filtre2 = data.age < 30

print(data[filtre1 & filtre2]) # cinsiyeti erkek ve yasi 20 den kucuk olanlari alir.
f,ax = plt.subplots(figsize=(14,14))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.age.plot(kind = 'line', color = 'g',label = 'age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.oldpeak.plot(color = 'r',label = 'oldpeak',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='sex', y='oldpeak',alpha = 0.5,color = 'red')

plt.xlabel('sex')              # label = name of label

plt.ylabel('oldpeak')

plt.title('sex Defense Scatter Plot')            # title = title of plot

plt.show()
data.age.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
for index,value in data[['thalach']][0:1].iterrows():

    print(index," : ",value)