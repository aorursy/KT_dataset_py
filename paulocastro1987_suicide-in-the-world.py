import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import kruskal,mannwhitneyu,ttest_ind

import os

os.chdir('/kaggle/input/')
df = pd.read_csv('master.csv')
df.head()
df.info()
df.columns #checking the columns
df.drop(['HDI for year','country-year',' gdp_for_year ($) '],axis=1,inplace=True)
df.head()
df['year'].unique() #unique years
plt.figure(figsize=(10,6))

sns.barplot(x = 'generation',y='suicides/100k pop',data = df,hue = 'sex')

plt.title('Profile of suicides by generation and sex')

plt.show()
plt.figure(figsize=(10,6))

sns.set()

plt.title('Profile of suicides by age and sex')

sns.barplot(x = 'age',y = 'suicides/100k pop',data = df,hue='sex')

plt.show()
x = df.groupby(['sex'])['suicides/100k pop'].mean()

y = df.groupby(['generation'])['suicides/100k pop'].mean()

z = df.groupby(['sex'])['suicides/100k pop'].std()
y #mean of suicides by generation
x #mean of suicides by sex
z #standard deviation by sex. It's really big.
sns.set()

plt.figure(figsize=(9,7))

plt.title('Profile of suicide/100k pop and sex')

k = sns.boxplot(x='sex',y = 'suicides/100k pop', data = df)

k.set_xticklabels(k.get_xticklabels(), rotation=45)

plt.show()
xy = df[df['sex'] == 'male'].iloc[:,4].values

xx = df[df['sex'] == 'female'].iloc[:,4].values
stat, p = mannwhitneyu(xx,xy)
p
ttest_ind(xx,xy)[1]
sns.set()

plt.figure(figsize=(9,7))

plt.title('Profile of suicide/100k pop and year')

k = sns.boxplot(x='age',y = 'suicides/100k pop', data = df)

k.set_xticklabels(k.get_xticklabels(), rotation=45)

plt.show()
z1 = df[df['age'] == '15-24 years'].iloc[:,4].values

z2 = df[df['age'] == '25-34 years'].iloc[:,4].values

z3 = df[df['age'] == '35-54 years'].iloc[:,4].values

z4 = df[df['age'] == '55-74 years'].iloc[:,4].values

z5 = df[df['age'] == '75+ years'].iloc[:,4].values
s,p = kruskal(z1,z2,z3,z4,z5)
p
a = df.groupby('year')['suicides/100k pop'].mean()
x = a.index

y = a.values
plt.figure(figsize=(9,6))

plt.title('Mean of suicides over world in the years')

plt.ylabel('Mean of suicides')

plt.xlabel('Year')

plt.plot(x,y)
plt.figure(figsize=(9,6))

sns.scatterplot(x = 'suicides/100k pop',y = 'gdp_per_capita ($)',data = df)

plt.title('Suicides and GDP')

plt.show()
a = df.groupby(['country'])['suicides/100k pop'].mean()
b = a.sort_values().tail(10)

b
plt.figure(figsize=(9,6))

plt.title('Mean of suicides index by country: 1987 - 2016')

plt.ylabel('Mean of suicides')

b.plot.bar()
c = (df.groupby(['year','country'])['suicides/100k pop'].mean())
c
d = df[df['year'] == 1985] #First year in data set

e = df[df['year'] == 1989] #Union of West Germany and East Germany 

f = df[df['year'] == 1991] #End of Soviet Union

g = df[df['year'] == 1995] #First crisis of neoliberalism in Russian Federation, Brazil, Argentina and Mexico.

h = df[df['year'] == 2008] #Crisis of capitalism

j = df[df['year'] == 2016] #Last year in data set
k = [d,e,f,g,h,j]

m = [1985,1989,1991,1995,2008,2016]

c = ['b','k','y','r','g','m']

for i in range(len(k)):

    aa = k[i].groupby(['country'])['suicides/100k pop'].mean()

    bb = aa.sort_values().tail(10)

    plt.figure(figsize=(9,6))

    plt.title('Top 10: Mean of suicides index by country: '+str(m[i]))

    plt.ylabel('Mean of suicides')

    bb.plot.bar(color=c[i])

    plt.show()
k = [d,e,f,g,h,j]

m = [1985,1989,1991,1995,2008,2016]

c = ['b','k','y','r','g','m']

for i in range(len(k)):

    aa = k[i].groupby(['country'])['suicides/100k pop'].mean()

    bb = aa.sort_values().head(5)

    plt.figure(figsize=(9,6))

    plt.title('Lower 5: Mean of suicides index by country: '+str(m[i]))

    plt.ylabel('Mean of suicides')

    bb.plot.bar(color=c[i])

    plt.show()