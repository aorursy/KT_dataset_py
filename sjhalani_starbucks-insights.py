import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

%matplotlib inline

sns.set(style="white", context="talk")



starbucks = pd.read_csv('../input/directory.csv')
#know the data

starbucks.head(10)

#rows and columns 

#2600 rows 

#13 columns

starbucks.shape
#checking of completeness of data

starbucks.notnull().sum()

#or 

starbucks.count()
#Checking for completeness percentage

starbucks.notnull().sum() * 100/starbucks.shape[0]
#Number of stores around the world

starbucks.count()[0]
#Number of country having starbucks

starbucks[starbucks.Country.duplicated()== False].count()[-1]

#or

len(list(set(starbucks.Country)))

#or

len(starbucks.Country.unique())
#Country having maximun number of stores

starbucks.groupby(['Country']).Country.count().sort_values(ascending=False)

#or 

starbucks.Country.value_counts().head(10)
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title = "Top 10 Countries with Most Number of Starbucks Stores")

starbucks.Country.value_counts().head(10).plot(kind="bar", color = "maroon")
#City with maximum stores

starbucks.groupby(['City']).City.count().sort_values(ascending=False)

#or

starbucks.City.value_counts().head(10)

fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title = "Top 10 Countries with Most Number of Starbucks Stores")

starbucks.City.value_counts().head(10).plot(kind="bar", color = "Blue")
#Ownership

starbucks[starbucks['Ownership Type'].duplicated() == False]['Ownership Type']

#or

starbucks['Ownership Type'].value_counts()

fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title = "Who owns the stores?")

starbucks['Ownership Type'].value_counts().plot(kind="bar", color = "maroon")

plt.show()
#number of stores in USA

usa_states = starbucks[starbucks['Country'] == 'US']

usa_states['State/Province'].value_counts().head(10)
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title="What are the Top 10 States in USA with most number of stores?")

usa_states['State/Province'].value_counts().head(10).plot(kind="bar")

plt.show()

#Brands under Starbucks

starbucks.Brand.value_counts()
fig = plt.figure(figsize=(8,5))

ax = fig.add_subplot(111)

ax.set(title="Brand under which Starbucks Operates")

starbucks.Brand.value_counts().plot(kind="bar", color = "maroon")

plt.show()