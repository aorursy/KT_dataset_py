# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df.info()
# correlation map

f,ax = plt.subplots(figsize=(8,8))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
df.head()
df.columns
df_suicide = (df.suicides_no / 1000)+ 2000



df_suicide.plot(kind='line', label='Suicide No', linewidth=2, alpha=.5, grid=True, linestyle=':')

df.year.plot(label='Year', linewidth=2, alpha=.5, grid=True, linestyle='-.')



plt.legend(loc='upper left')

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Line Plot')

plt.show()
df.plot(kind='scatter', x='population', y='suicides_no', alpha=.5)

plt.xlabel('Population')

plt.ylabel('Suicide No\'s')

plt.title('Population - Suicide No Scatter Plot')

plt.show()
df['suicides/100k pop'].plot(kind='hist',bins=10, density=1)
myFilter = (df.year > 2010) & (df['suicides/100k pop'] > 100)

df[myFilter]
#user defined function

def square(x):

    return x**2

print(square(4))
#scope

x=5

y = 4

def scopeExample():

    x=1

    return x*y

print(x)

print(scopeExample())
# default args

def calculatePower(x, y=2):

    return x**y

print(calculatePower(2))

print(calculatePower(2,3))
def expArgs(*args):

    return [i for i in args]

print(expArgs(1,2,3,4,5))
square = lambda x: x**2

print(square(2))
n = 'someone'

iterableN = iter(n)

print(next(iterableN))

print(*iterableN)
df.head()
threshold = sum(df['gdp_per_capita ($)'])/len(df['gdp_per_capita ($)'])

df['average'] = ['higher than average' if i > threshold else 'lower than average' for i in df['gdp_per_capita ($)']]

df.head()
df.shape
df.columns
df.info()
print(df['country'].value_counts(dropna =False))
df.describe()
df.boxplot(column='HDI for year')

plt.show()
df_ = df.head(10)

df_
melted = pd.melt(frame=df_,id_vars = 'generation', value_vars='suicides_no')

melted
melted = melted.head(2)

melted.pivot(index = 'generation', columns = 'variable',values='value')
df1 = df.head(10)

df2 = df.tail(10)

conc_df = pd.concat([df1,df2], axis=0, ignore_index=True)

conc_df
df1 = conc_df.population

df2 = conc_df.suicides_no

concat_df = pd.concat([df1,df2],axis=1)

concat_df
df.dtypes
df_copy = df.copy()

df_copy['year'] = df['year'].astype('object')

df_copy.dtypes
df.info()
df['HDI for year'].value_counts(dropna=False)
df_copy['HDI for year'].dropna(inplace=True)
assert df_copy['HDI for year'].notnull().all()

# Will return nothing if we successfully dropped null values.
df_copy['HDI for year'].value_counts(dropna=False)
country = ['Norway', 'Sweden', 'Denmark', 'Finland', 'Netherlands']

capital = ['Oslo', 'Stockholm', 'Copenhagen', 'Helsinki', 'Amsterdam']

label = ['country', 'capital']

list_col= [country, capital]

zipped = dict(zip(label, list_col))

new_df = pd.DataFrame(zipped)

new_df
new_df['continent']= 'Europe'

new_df
new_df['hdi_ranking'] = [1,7,11,15,10]

new_df
df1 = df.loc[:,['suicides_no', 'suicides/100k pop']]

df1.plot()

plt.show()
df1.plot(subplots=True)

plt.show()
df1.plot(kind='scatter', x='suicides_no', y='suicides/100k pop')

plt.show()
df1.plot(kind="hist", y="suicides_no", bins=50, range=(0,250), normed = True)

plt.show()
fig,axes = plt.subplots(nrows=2,ncols=1)

df1.plot(kind='hist', y='suicides/100k pop', bins=50, range=(0,100), normed=True, ax=axes[0])

df1.plot(kind='hist', y='suicides/100k pop', bins=50, range=(0,100), normed=True, ax=axes[1], cumulative=True)

plt.savefig('graph.png')

plt
df.describe()
time_list =  ['1992-03-08']

print(type(time_list[0]))

dto = pd.to_datetime(time_list)

print(type(dto))
df2 = df.head()

dList = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

dto = pd.to_datetime(dList)

df2['date'] = dto



df2 = df2.set_index('date')

df2
print(df2.loc["1993-03-16"])

print(df2.loc["1992-01-10":'1993-03-15'])
df2.resample('A').mean() #'M' = month, 'A'=year
df2.resample('M').mean()
df2.resample('M').first().interpolate('linear')
df2.resample('M').mean().interpolate('linear')

# now we will interpolate with mean unlike the first one which just fill the gap between two null data
df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
df.loc[1,['age']]
df[['sex', 'age']]
print(type(df['age']))

print(type(df[['age']]))
df.loc[1:10,'year':'age']
ff = df.suicides_no > 10000

sf = df.year > 2010

df[ff & sf]
df.year[df.suicides_no>11000]
def div(n): return n/2

df['gdp_per_capita ($)'].apply(div)
df['gdp_per_capita ($)'].apply(lambda n : n/2)
df['no_sense'] = df.year / df.suicides_no

df.head()
print(df.index.name)

df.index.name = 'index'

df.head()
df3 = df.copy()

df3.index = range(100,27920)

df3.head()
df3 = df3.set_index(['population', 'suicides_no'])

df3.head()
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df4 = pd.DataFrame(dic)

df4
df4.pivot(index='treatment', columns='gender',values='response')
df5 = df4.set_index(['treatment','gender'])

df5
df5.unstack(level=0)
df5.unstack(level=1)
df5 = df5.swaplevel(0,1)

df5
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df4,id_vars="treatment",value_vars=["age","response"])
df4.groupby('treatment').mean()
df4.groupby('treatment').age.max()
df4.groupby('treatment')[['age', 'response']].min()