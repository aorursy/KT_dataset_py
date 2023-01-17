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
data = pd.read_csv('../input/master.csv')

data.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data.columns
data.shape
data.isnull().sum()
data.isnull().mean()
data = data.drop(['HDI for year', 'country-year'], axis = 1)
data.columns
data = data.rename(columns = {'country':'Country', 'year':'Year', 'sex':'Sex', 'age':'Age', 'suicides_no':'Suicides', 'population':'Population', 

                             'suicides/100k pop':'SuicideRate', ' gdp_for_year ($) ':'GDP',

                              'gdp_per_capita ($)': 'GDPperCapita', 'generation': 'Generation'})
data.columns
data.describe()
data.info()
data['Country'].unique()
#Number of Countries in Dataset

len(data['Country'].unique())
data['Sex'].unique()
data['Age'].unique()
data['GDP'].unique()
data['Generation'].unique()
data.corr()
sns.heatmap(data.corr())
data_year = data.groupby('Year')
plt.figure(figsize= (15,5))

plt.bar(x = data_year['Suicides'].sum().keys(), height = data_year['Suicides'].sum())

plt.show()
plt.figure(figsize= (15,5))

plt.bar(x = data_year['SuicideRate'].mean().keys(), height = data_year['SuicideRate'].mean())

plt.show()
data_country = data.groupby('Country')
plt.figure(figsize=(20,10))

height = 100*data_country['Suicides'].sum()/data['Suicides'].sum()

x = data_country['Suicides'].sum().keys()

plt.bar(x = x, height= height)

plt.xticks(rotation='vertical')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center', rotation='vertical', va='bottom')

plt.show()
height['Japan'] + height['Russian Federation'] + height['United States']
plt.figure(figsize=(20,10))

height = data_country['SuicideRate'].mean()

x = data_country['SuicideRate'].mean().keys()

plt.bar(x = x, height= height)

plt.xticks(rotation='vertical')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center',rotation='vertical',va='bottom')

plt.show()
data_gender = data.groupby('Sex')
plt.figure(figsize=(10,4))

height = 100*data_gender['Suicides'].sum()/data['Suicides'].sum()

x =  data_gender['Suicides'].sum().keys()

plt.bar(x = x, height = height)

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center', fontweight='bold')

plt.show()
plt.figure(figsize=(10,4))

height = data_gender['SuicideRate'].mean()

x =  data_gender['SuicideRate'].mean().keys()

plt.bar(x = x, height = height)

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')

plt.show()
data_age = data.groupby('Age')
plt.figure(figsize=(10,4))

height = 100*data_age['Suicides'].sum()/data['Suicides'].sum()

x =  data_age['Suicides'].sum().keys()

plt.bar(x = x, height = height)

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')

plt.show()
plt.figure(figsize=(10,4))

height = data_age['SuicideRate'].mean()

x =  data_age['SuicideRate'].mean().keys()

plt.bar(x = x, height = height)

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')

plt.show()