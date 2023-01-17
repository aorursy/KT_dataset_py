# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
#Read data
data = pd.read_csv('../input/train.csv')
data.columns
data.columns.size
data.shape
data.columns = [column.lower() for column in data.columns]
data.columns
#I could not understand what some columns name represents so I changed them.
data = data.rename(
    columns = {
        'passengerId': 'id',
        'pclass': 'ticketClass',
        'sibsp': 'sisBroSpo',
        'parch': 'parentChild'
    }
)
#More understandable.
data.columns
data.dtypes
data.info()
data.isnull().sum()
data.describe()
#From output of describe method above it looks like;
#More people were travelling with ticketClass 3 then other groups
#Most people have no sisBroSpo(sibsb) aboard
#Most people have no parentChild(parch) aboard
#sisBroSpo and parentChild could be connected?
data.plot()
plt.show()
data.head()
data.tail()
data.boxplot(column = ['age', 'sisBroSpo', 'parentChild'])
plt.show()
data.plot(kind = 'scatter', x = 'sisBroSpo', y = 'parentChild')
plt.show()
#Ticket class number's
print(data.ticketClass.unique())
print(data.ticketClass.value_counts())
pd.melt(
    frame = data, 
    id_vars = 'age', 
    value_vars = ['sisBroSpo', 'parentChild']
).head(10)
pd.melt(
    frame = data, 
    id_vars = 'age',
    value_vars = ['sisBroSpo', 'parentChild']
).tail(10)
filtered = data[(data.age >= 1) & (data.survived == 1)]
filtered.loc[filtered.age == filtered.age.max()]
filtered = data[(data.age >= 1) & (data.survived == 0)]
filtered.loc[filtered.age == filtered.age.max()]
filtered = data[(data.age <= 1) & (data.survived == 1)]
filtered.loc[filtered.age == filtered.age.min()]
filtered = data[(data.age <= 1) & (data.survived == 0)]
filtered.loc[filtered.age == filtered.age.min()]
#data.loc[:,['sisBroSpo', 'parentChild']].plot()
data.loc[:,['sisBroSpo', 'parentChild']].plot(subplots = True)
plt.legend(loc = 'upper right')
plt.show()
data[data.age > 50].plot()
plt.show()
data[data.age > 50].plot(subplots = True)
plt.show()
data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'parentChild',
    color = 'yellow'
)
plt.show()
data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'sisBroSpo',
    color = 'yellow'
)
plt.show()
data[data.age < 50].plot()
plt.show()
data[data.age < 50].plot(subplots = True)
plt.show()
data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'parentChild',
    color = 'yellow'
)
plt.show()
data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'sisBroSpo',
    color = 'yellow'
)
plt.show()
data.sort_values('age').age.unique()
keep = ['name', 'age', 'sisBroSpo', 'parentChild', 'survived']
ageAboveFifty = data[data.age > 50].filter(items = keep).head()
ageBelowFifty = data[data.age < 50].filter(items = keep).head()
ageAboveFifty = ageAboveFifty.sort_values('age')
ageAboveFifty
ageBelowFifty = ageBelowFifty.sort_values('age')
ageBelowFifty
ageAboveFifty.where(ageAboveFifty.values != ageBelowFifty.values)
data.describe()
#Find percentage of people surviving if they are above 28
ageAbove28 = data[
    (data.age >= 28) & (data.survived == 1)
].age.count() / data[(data.survived == 1)].age.count()
#Find percentage of people surviving if they are below 28
ageBelow28 = data[
    (data.age < 28) & (data.survived == 1)
].age.count() / data[(data.survived == 1)].age.count()
ticketOne = data[
    (data.ticketClass == 1) & (data.survived == 1)
].ticketClass.count() / data[(data.survived == 1)].ticketClass.count()
ticketTwo = data[
    (data.ticketClass == 2) & (data.survived == 1)
].ticketClass.count() / data[(data.survived == 1)].ticketClass.count()
ticketThree = data[
    (data.ticketClass == 3) & (data.survived == 1)
].ticketClass.count() / data[(data.survived == 1)].ticketClass.count()
fareAbove = data[
    (data.fare >= 14) & (data.survived == 1)
].fare.count() / data[(data.survived == 1)].fare.count()
fareBelow = data[
    (data.fare < 14) & (data.survived == 1)
].fare.count() / data[(data.survived == 1)].fare.count()
pSurvivor = data[(data.survived == 1)].passengerid.count() / data.passengerid.count()
pSurvivor
