# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("../input/heart.csv") #firstly reads csv formatting data 
data.columns
data.info()
data.head() #first 5 data shows 
data.tail() #last 5 data shows
data.describe()
data.dtypes
data.corr() #for correlation
data.isnull().values.any() #all rows control for null values.İf trouble is False,it's no problem
plt.figure(figsize=(14,14))

sns.heatmap(data.corr(),annot=True,fmt='0.1f')

plt.show()
data["age"][0:10]>60 
x=data["age"][0:10]

x
data[(data['age']>65) & (data['sex']>0) & (data["chol"]<270)]
data.age.value_counts()[:10] #frequency
sns.barplot(x=data.age.value_counts()[:10].index,y=data.age.value_counts()[:10].values)

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
minAge=min(data["age"])

maxAge=max(data["age"])

meanAge=data.age.mean() #or data["age"].mean() that's same

print("Min Age in Data:",minAge)

print("Max Age in Data:",maxAge)

print("Mean Age in Data",meanAge)
young_ages=data[(data.age>=29)&(data.age<40)]

middle_ages=data[(data.age>=40)&(data.age<70)]

old_age=data[(data.age>70)]

print("Genç Kesim:",len(young_ages),"kişi")

print("Orta Yaşlı:",len(middle_ages),"kişi")

print("yaşlılar:",len(old_age),"kişi")

sns.barplot(x=['young_ages','middle_ages','old_age'],y=[len(young_ages),len(middle_ages),len(old_age)])

plt.xlabel('Age Range')

plt.ylabel('Age Counts')

plt.show()