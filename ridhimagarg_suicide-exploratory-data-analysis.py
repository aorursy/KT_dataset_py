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



import matplotlib.pyplot as plt

import seaborn as sns

import re
data = pd.read_csv('../input/master.csv')
data.head()
data.isna().sum()
data['country'].unique()
len(data['country'].unique())
data['country'].value_counts()
data['country_year'] = data['country-year'].apply(lambda x: int(re.findall("[0-9]+",x)[0]))
data['country_year']
## checking if all years and country-year are equal

set(data['year'] == data['country_year'])
data = data.drop(['country_year'], axis=1)
data.columns
data['age'] = data['age'].replace(regex = {"years":''})
data['age']
data['age'].unique()
data['sex'].unique()
data['generation'].unique()
data['generation'].value_counts()
data[(data['country'] == 'Albania') & (data['year'] == 1987)].sort_values(by='age')
country_suicide = data.groupby('country').agg('mean')['suicides_no'].sort_values(ascending=False)
country_suicide.values
x = list(country_suicide.keys())

y = list(country_suicide.values)

plt.figure(figsize=(12,16))

plt.barh(x,y)

plt.show
plt.figure(figsize = (8,6))

x = list(data.groupby(['year']).agg('sum')['suicides_no'].keys())

y = list(data.groupby(['year']).agg('sum')['suicides_no'].values)

plt.title("Yearly suicides rate")

plt.xlabel("Years")

plt.ylabel("suicide no.")

plt.bar(x, y)

#plt.set_xticklabels(tic)

plt.show()
sns.lmplot(x="year", y="suicides_no", data=data)
data.groupby('country').agg(['min','max'])['suicides_no'].sort_values(by='max', ascending=False)
fig, ax = plt.subplots(figsize=(16,16))

country_year_average = data.groupby(['year', 'country']).agg('mean')['suicides_no']

country_year_average.unstack().plot(ax=ax)

country_year_average  = country_year_average.reset_index()

country_year_average = country_year_average[country_year_average['suicides_no']>1000]
country_year_average
country_year_average.sort_values(by='suicides_no', ascending=False)
country_year_average['year']
# fig, ax = plt.subplots(figsize=(16,16))

# data.groupby(['country', 'year']).agg('mean')['suicides_no'].unstack().plot(kind='violin',ax=ax)



#country_year_average = data.groupby(['country', 'year']).agg('mean')['suicides_no']

#plt.subplots(figsize=(8,8), )

plt.figure(figsize=(8,8))

g = sns.FacetGrid(country_year_average, col='year', height= 6, size=4, col_wrap=5)

g.map(plt.bar, "country", "suicides_no")

g.set_xticklabels(rotation=30,fontsize=10)
plt.figure(figsize=(4,8))

x = list(data.groupby(['age']).agg('sum')['suicides_no'].keys())

y= list(data.groupby(['age']).agg('sum')['suicides_no'].values)

plt.figure(figsize=(4,4))

plt.barh(x,y)

plt.show
sns.countplot(x='sex', data=data)
count_sex_total = data.groupby(['country','sex']).agg('sum')['suicides_no'].reset_index()



plt.figure(figsize=(8,8))

g = sns.FacetGrid(count_sex_total, col="country", height=6, size=4, col_wrap=6)

g.map(plt.bar, "sex", "suicides_no")

plt.show()
plt.figure(figsize=(8,4))

x = list(data.groupby(['sex']).agg('sum')['suicides_no'].keys())

y= list(data.groupby(['sex']).agg('sum')['suicides_no'].values)

sns.barplot(x, y, data=data)

plt.show()
age_gender_suicides_sum = data.groupby(['sex','age']).agg('sum')['suicides_no'].reset_index()
age_gender_suicides_sum
plt.figure(figsize=(8,8))

g = sns.FacetGrid(age_gender_suicides_sum, col='age', height= 6, size=4, col_wrap=3)

g.map(plt.bar, "sex", "suicides_no")

g.set_xticklabels(rotation=30,fontsize=10)
plt.figure(figsize=(8,6))

sns.countplot(x='generation', data=data, order = data['generation'].value_counts().index)

plt.show()
plt.figure(figsize=(8,6))

sns.countplot(x='generation', data=data, order = data['generation'].value_counts().index, hue='age')

plt.show()
data.groupby(['generation']).agg(['sum'])['suicides_no']
plt.figure(figsize=(8,4))

x = list(data.groupby(['generation']).agg('sum')['suicides_no'].keys())

y= list(data.groupby(['generation']).agg('sum')['suicides_no'].values)

sns.barplot(x, y, data=data)

plt.show()
generation_gender_sum = data.groupby(['generation', 'sex']).agg('sum')['suicides_no'].reset_index()



plt.figure(figsize= (12,4))

g = sns.FacetGrid(generation_gender_sum, col='sex', height=6, size=4)

g.map(plt.bar, "generation", "suicides_no")

g.set_xticklabels(rotation=90)
data.columns
for e in set(data['country']):

    plt.figure(figsize= (8,8))

    ax = sns.barplot(x="country-year", y="gdp_per_capita ($)", data=data[data['country']==e])

    for label in ax.xaxis.get_ticklabels():

        label.set_rotation(30)

    plt.show()

    