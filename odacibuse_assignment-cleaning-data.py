# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

data.head()

data.tail()
data.columns
data.describe
data.info()
data['HDI for year'].value_counts(dropna=False)  # with dropna=False, we get NaN values data

data['HDI for year'].dropna(inplace=True)  # we updated data after drop values

# We can check with assert statement

assert data['HDI for year'].notnull().all()  # returns nothing because we drop nan values

# We changed column names to lowercase

data.columns = [each.lower() for each in data.columns]

# added missed _ to column names

data.columns = [each.split()[0] + '_' + each.split()[1] + '_' + each.split()[2] if (len(each.split()) > 2) else each for

                each in data.columns]

data.columns

data.plot(kind='line', x='year', y='suicides_no')

# plt.xticks(rotation=90) # can change x axis's labels to vertical 

plt.title('Global suicide number')

plt.show()

labels = 'Male', 'Female'

sizes = data.sex.value_counts()

print(sizes)

explode = (0.05, 0)

colors = ['lightskyblue', 'pink']

plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=90)  # with startangle we turned chart 90

plt.title('Suicides by Sex')

plt.legend(labels, loc='upper right')

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 9))

dataFemale = data[(data.sex == 'female')]

dataMale = data[(data.sex == 'male')]

# with hue='age'; we grouped data by age

sns.lineplot(x='year', y='suicides_no', hue='age', color='pink', data=dataFemale, ax=ax[0], marker="o").set_title(

    "Female' s suicides by years", fontsize=20)

sns.lineplot(x='year', y='suicides_no', hue='age', color='lightskyblue', data=dataMale, ax=ax[1], marker="o").set_title(

    "Male's suicides by years", fontsize=20)

fig.show()

# countplot, show the counts of observations in each categorical bin using bars.

plt.figure(figsize=(10, 25))

sns.countplot(y='country', data=data)

plt.title('Data by country')

plt.show()

melted = pd.melt(frame=data, id_vars='country', value_vars=['suicides_no'])

melted

# lets create 2 data frame

data1 = data.head()

data2 = data.tail()

concatData = pd.concat([data1, data2], axis=0, ignore_index=True)  # axis=0, we add row to each other

concatData

# We can also add column to each other

data1 = data['year'].head()

data2 = data['suicides_no'].head()

concatData = pd.concat([data1, data2], axis=1)

concatData
