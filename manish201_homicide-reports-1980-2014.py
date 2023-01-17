# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

mydata=pd.read_csv("..//input/database.csv",low_memory=False)
#drop the unnecessary columns

mydata=mydata.drop(['Record ID','Agency Code','Perpetrator Count','Victim Count','Record Source'],axis=1)
pd.isnull(mydata)#checks for Null values
#records showing victime age<15

mydata[mydata['Victim Age']<15]
## Rate of crime's solved

solved = pd.DataFrame(mydata, columns = ['Crime Solved']) 

resolution = solved.stack().value_counts()

ax = resolution.plot(kind = 'pie',

                              title = 'Rates of crimes solved between 1980 & 2014 (in %)',

                              startangle = 10,

                              autopct='%.2f')

ax.set_ylabel('')
## Rate of crime type

crimetype = pd.DataFrame(mydata, columns = ['Perpetrator Sex']) 

resolution = crimetype.stack().value_counts()

ax = resolution.plot(kind = 'pie',

                              title = 'Rates of crime types (in %)',

                              startangle = 10,

                              autopct='%.2f',fontsize=20)

ax.set_ylabel('')
#Crimes by State

state = pd.DataFrame(mydata, columns = ['State']) 

count_states = state.stack().value_counts()

states = count_states.sort_index(axis=0, ascending=False)

#plot the total of homicides

print(states.plot(kind='barh', fontsize=10,  width=0.5,  figsize=(10, 10), title='Homicides by State between 1980 and 2014'))

#count of crime solved

crimesolved = pd.DataFrame(mydata, columns = ['Crime Solved']) 

count_solved = crimesolved.stack().value_counts()

count_solved
#count of weapon used

sns.countplot(y="Weapon",data=mydata, palette="Greens_d");
#crime solved by agency

sns.countplot(y="Agency Type", hue="Crime Solved", data=mydata,saturation=1)