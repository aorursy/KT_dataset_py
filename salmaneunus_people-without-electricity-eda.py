# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
e = pd.read_csv("../input/number-of-people-without-electricity/people-without-electricity-country.csv" )
e.head()
e

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plt.title("Change in the number of people without electricity over time")

sns.lineplot(x="Year",y = "Number of people without access to electricity (people without electricity access)",data=e)

plt.xlabel("Years")

plt.ylabel("Number of people without Electricity")
plt.figure(figsize=(10,5))

sns.boxplot('Number of people without access to electricity (people without electricity access)', data=e)
e.describe()
e.tail()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(34,20))

plt.title("Change in the number of people without electricity in different countries")

sns.lineplot(x="Code",y = "Number of people without access to electricity (people without electricity access)",data=e)

plt.xlabel("Different Countries")

plt.ylabel("Number of people without Electricity")
m = pd.read_csv("../input/number-of-people-without-electricity/people-without-electricity-country.csv" ,index_col = "Entity")
plt.figure(figsize=(21,14))

plt.title("Comparison of the number of people without electricity in various countries")

sns.barplot(x = m.index,y=m['Number of people without access to electricity (people without electricity access)'])

plt.xlabel("Countries")

plt.ylabel("Change in number of people without electricity")

j = pd.read_csv("../input/number-of-people-without-electricity/people-without-electricity-country.csv" ,index_col = "Year")
plt.figure(figsize=(14,7))

sns.scatterplot(x=e['Year'],y=e['Number of people without access to electricity (people without electricity access)'])
plt.figure(figsize=(14,7))

sns.regplot(x=e['Year'],y=e['Number of people without access to electricity (people without electricity access)'])
plt.figure(figsize=(30,20))

sns.scatterplot(x=e['Year'],y=e['Number of people without access to electricity (people without electricity access)'],hue = e['Entity'])

z = e.iloc[400,0:].describe
sns.lmplot(x="Year",y="Number of people without access to electricity (people without electricity access)",hue="Code",data=e)
plt.figure(figsize=(14,7))

sns.distplot(a=m['Number of people without access to electricity (people without electricity access)'],kde=False)
plt.figure(figsize=(14,7))

sns.distplot(a=j['Number of people without access to electricity (people without electricity access)'],kde=False)
plt.figure(figsize=(40,40))

sns.swarmplot(x="Year",y="Number of people without access to electricity (people without electricity access)",data=e)
sns.kdeplot(data=e['Number of people without access to electricity (people without electricity access)'],label="Number of people without Electricity",shade=True) 
sns.kdeplot(data=m['Number of people without access to electricity (people without electricity access)'],label="Number of people without Electricity",shade=True) 
plt.figure(figsize=(40,40))

sns.swarmplot(x="Year",y="Number of people without access to electricity (people without electricity access)",data=m)
e.Year.dtype
plt.figure(figsize = (20,20))

df = e[(e['Number of people without access to electricity (people without electricity access)']>=5000) & (e['Number of people without access to electricity (people without electricity access)']<8000)]

sns.boxplot('Year', 'Number of people without access to electricity (people without electricity access)', data=df)


import numpy as np

plt.figure(figsize = (20,20))

# get correlation matrix

corr = e.corr()

fig, ax = plt.subplots()

# create heatmap

im = ax.imshow(corr.values)



# set labels

ax.set_xticks(np.arange(len(corr.columns)))

ax.set_yticks(np.arange(len(corr.columns)))

ax.set_xticklabels(corr.columns)

ax.set_yticklabels(corr.columns)



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")
plt.figure(figsize = (20,20))

sns.heatmap(e.corr(), annot=True)
plt.figure(figsize = (60,30))

sns.pairplot(e)