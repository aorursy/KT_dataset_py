# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

df = pd.read_csv('/kaggle/input/russian-demography/russian_demography.csv')
df.head()
df.info()
df.isnull().sum()
df.dtypes
df['region'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df.head()
df.shape
##Let's check average birth and death rates over the years
df.groupby(['year']).agg({'death_rate':'mean'}).plot.bar(figsize=(20,10))

plt.show()
df.groupby(['year']).agg({'birth_rate':'mean'}).plot.bar(figsize=(20,10))

plt.show()
df['rate_diff'] = df['birth_rate'] - df['death_rate']
##Plotting the difference in the birth and death rate, which will show us an actual increase/decrease in the population

#over the years
df.groupby(['year']).agg({'rate_diff':'mean'}).plot.bar(figsize=(20,10))

plt.show()
#After 1991, Russia's population has been on a huge decline.

##From 2012 to 2016 it started increasing a bit, but 2017 showed a decrease yet again
df.head()
#Now let's check the natural population growth per 1000 people
df.groupby(['year']).agg({'npg':'mean'}).plot.bar(figsize=(20,10))

plt.show()
##So, npg is same as rate_diff
#Let's check the same stats filtered by region
df.groupby(['region']).agg({'npg':'mean'}).plot.bar(figsize=(20,10))

plt.show()
df.groupby(['region']).agg({'npg':'mean'}).sort_values(by='npg',ascending=False).plot.bar(figsize=(20,10))
##We can where the Population has increased in Russia, and where the population has decreased exteremely

#In Chenchen Republic, the population has increased extensively, whereas in Pskov Oblast, it has decreased exteremely

#A few places like Kamchakta Krai have had almost no effect of the change in population
df.head()
df.groupby(['region']).agg({'migratory_growth':'mean'}).sort_values(by='migratory_growth',ascending=False).plot.bar(figsize=(20,10))
df['migratory_growth'].isna().sum()/df.shape[0]
##Since migratory has very few null values, let's check for all variables
df.isnull().sum()/df.shape[0]
#Only migratory growth has so many nulls

#let's check how migratory growth is related to npg, then we can take a decision, to impute or remove nulls
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
df = df.drop(columns='rate_diff')
##Death rate and NPG have high negative correlation, and that is perfectly valid
plt.figure(figsize=(20,10))

sns.scatterplot(data=df,x='npg',y='migratory_growth')
##As per the above, we can actually impute migratory growth with the some value
df['migratory_growth'].median()
#Median imputation seems fine for now
df['migratory_growth'].fillna(df['migratory_growth'].median(),inplace=True)
#Checking plots again
df.groupby(['region']).agg({'migratory_growth':'mean'}).sort_values(by='migratory_growth',ascending=False).plot.bar(figsize=(20,10))
##We see that the plot has changed, and it does show some obvious truths, because we understand that it is quite normal

#for people to shift to Moscow, it being the capital
plt.figure(figsize=(20,10))

sns.scatterplot(data=df,x='npg',y='migratory_growth')
##Scatter plot still looks almost the same, which means the pattern was preserved
#Let's check nulls again
df.isnull().sum()
##Also, we could one more thing, since the data is based on time
#We can impute using methods such as forward fill, so we will be doing that now
df = pd.read_csv('/kaggle/input/russian-demography/russian_demography.csv')
df.isnull().sum()
df.fillna(method='ffill',inplace=True)
df.isnull().sum()
##We will have to backfill for migratory_growth, because it starts with null itself
df = pd.read_csv('/kaggle/input/russian-demography/russian_demography.csv')
df.isnull().sum()
df['migratory_growth'].fillna(method='bfill',inplace=True)
df.isnull().sum()
df.fillna(method='ffill',inplace=True)
##Now, our data should be good enough for initial analysis
df.head()
##We can do one more thing, that is fill null values as per the regions. We will try that a bit later
df.groupby(['region']).agg({'npg':'mean'}).sort_values(by='npg',ascending=False).plot.bar(figsize=(20,10))
##Now,this clearly shows that a few regions have been extensively occupied by people, and people have probably migrated from

#other regions, and isn't that very obvious in today's World?
##Let's check what happened over the years
df.groupby(['year']).agg({'npg':'mean'}).plot.bar(figsize=(20,10))

plt.show()
##So, it is true that Russia's population has been on a decline lately
##Let's use a pairplot, to try to find out what effects what!
sns.pairplot(data=df)
##We can see that migratory growth is basically a constant for most part of it, and quite obviously,

#npg is positively correlated to birth_rate and negatively to the death_rate