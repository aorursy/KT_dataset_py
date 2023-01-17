# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import pandas_profiling

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading the data

data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.info()
# describing the data



data.describe()
#correlation values

data.corr()


profile = pandas_profiling.ProfileReport(data)

profile
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
f,ax = plt.subplots(figsize=(10, 5))

sns.distplot(data['age'], color = 'g')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
size = data['sex'].value_counts()

colors = ['lightblue', 'lightgreen']

labels = "Male", "Female"

explode = [0, 0.01]



f,ax = plt.subplots(figsize=(7, 7))

plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')

plt.title('Distribution of Gender', fontsize = 20)

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x='age',data = data, hue = 'target',palette='GnBu')

plt.show()
#Line Plot

data.chol.plot(kind = 'line', color = 'g',label = 'chol',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')

data.thalach.plot(color = 'r',label = 'thalach',linewidth=1, alpha = 0.7,grid = True,linestyle = '-.')

plt.legend(loc='upper right')

plt.title('Relation of Cholestrol with Thalach', fontsize = 20)

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='chol',y='thalach',data=data,hue='target')

plt.show()
plt.figure(figsize=(8,6))

data.trestbps.plot(kind = 'line', color = 'g',label = 'trestbps',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')

data.thalach.plot(color = 'r',label = 'thalach',linewidth=1, alpha = 0.7,grid = True,linestyle = '-.')

plt.legend(loc='upper right')

plt.title('Relation of Trestbps with Thalach', fontsize = 20)

plt.show()
x = data['chol']>300 #kolestrolü 300 den fazla olanlar

data[x]
data[np.logical_and(data['chol']>300, data['age']<50 )] #Yaşı 50denküçük olup kolestrolü 300 den fazla olanlar
data[np.logical_and(data['chol']>300, data['age']>50 )].sex.plot(kind = 'hist',bins = 20,color='red',figsize = (10,10))



plt.show() #kolestrolü 300 den ve yaşı 50den büyük insanların cinsiyete göre dağılımı.