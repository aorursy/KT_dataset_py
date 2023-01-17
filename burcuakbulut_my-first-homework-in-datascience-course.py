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
import numpy as np

import pandas as pd

import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt 

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.
data = pd.read_csv('../input/heart.csv') #reading data for csv
print("Shape of this data {} \n{} is no of row and {} is no of column." .format(data.shape, data.shape[0], data.shape[1]))

data.head(5)

#if you want to show end of list you can data.tail





#data.tail() 
data.columns

#you can learn data's features
data.info()
data.plot(kind="scatter", x="age", y="chol", alpha= 0.5, color="red",grid=True, label = "age",linewidth=0.5 ) 

plt.legend(loc="upper right")

plt.ylabel("Chol")

plt.xlabel("Age (year)")

plt.title("Heart Diseases Related to Chol and Age")

plt.show()

#if you want to compare 'age' and 'chol', you can use Scatter Plot
#data.Speed.plot(kind='hist', bins=50, figsize=(50,50)), if you want, you can use this code. Similar codes.



data.age.hist(bins=50, figsize=(8,8), color= 'purple')

plt.xlabel('Age (year)', color='blue')

plt.ylabel('Trestbps (Hg)', color='blue')

plt.title('Corelation Age and Trestbps', color='blue')

plt.show()



#I want to compare 'age' and 'trestbps', is it related to each other?
data.hist(bins=50, figsize=(15,15), color= 'blue', grid=False)

plt.show()

#you can show all data of frequency.
# using the seaborn library.

sns.set(style="whitegrid")              

g = sns.jointplot("age", "oldpeak", data=data, kind="reg",xlim=(0, 60), ylim=(0, 12), color="Green")

plt.show()





data.plot(kind = "hist", bins = 14, figsize = (8,8), label = "all data", alpha=0.65)

plt.xlabel("all data",FontSize = 10)

plt.ylabel("Frequency",FontSize = 10)

plt.legend()

plt.show()
data.describe()



f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, cmap="BuPu", fmt='.1f', ax=ax)

plt.show()
x= ((data['age']>65) &(data['sex']==1))

data[x]

#list of age greater than 65 male

#data filter
y= ((data['age']>65) &(data['sex']==0))

data[y]

#list of age greater than 65 female

#data filter
data[data.sex==1].age.hist(bins=20, color='blue')

data[data.sex==0].age.hist(bins=20, color='purple')

plt.xlabel('Age(year)')

plt.ylabel('Populations')

plt.show()

#  If you want to filter, you can do it this way and see the statistical values of the data;

x= ((data['age']>45) &(data['sex']==1))

data[x].describe()
y= ((data['age']>45) &(data['sex']==0))

data[y].describe()
print(data[x], data[y])
#data visualization with point plot

f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x=data['age'],y=data['chol'],color='firebrick',alpha=0.8, label='chol')

sns.pointplot(x=data['age'],y=data['thalach'],color='purple',alpha=0.8, label='thalach')

plt.text(2,350,'chol',color='firebrick',fontsize=18,style='normal')

plt.text(2,335,'thalach',color='purple',fontsize=18, style= 'normal')

plt.xlabel('Ages',fontsize=15,color='blue')

plt.ylabel('Values',fontsize=15,color='blue')

plt.title('Chol vs Oldpeak Values',fontsize=20,color='blue')

plt.legend()

plt.show()
pd.plotting.scatter_matrix(data.loc[:,data.columns!='Target'],

                          c=['green','blue','red'], figsize=[15,15],diagonal='hist', alpha=0.8,

                          s=20, marker='o', edgecolor='black')

plt.show()
#find min and max ages



minAge=min(data.age)

maxAge=max(data.age)

meanAge=data.age.mean()

print('Min Age :',minAge)

print('Max Age :',maxAge)

print('Mean Age :',meanAge)

#classification ages



y_ages=data[(data.age>=29)&(data.age<40)]

m_ages=data[(data.age>=40)&(data.age<55)]

e_ages=data[(data.age>55)]

print('Young Ages :',len(y_ages))

print('Middle Ages :',len(m_ages))

print('Elderly Ages :',len(e_ages))



sns.barplot(x=['young ages','middle ages','elderly ages'], y=[len(y_ages),len(m_ages),len(e_ages)])

plt.xlabel('Age Range',size=15)

plt.ylabel('Counts',size=15)

plt.title('Ages State in Dataset',size=16)

plt.show()



male_count=len(data[data['sex']==1])

female_count=len(data[data['sex']==0])

print('Male Count    :',male_count)

print('Female Count  :',female_count)
sns.countplot(data.cp)

plt.xlabel('Chest Type')

plt.ylabel('Count')

plt.title('Chest Type vs Count ')

plt.show()



#condition level

#0 +

#1 ++

#2 ++++

#3 +++++++
chest_type=data[(data.cp == 3)&(data.age>60)]

print('high-risk:',chest_type)

            