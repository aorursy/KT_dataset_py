import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

import os



print(os.listdir("../input"))

df = (pd.read_csv("../input/master.csv"))

print(df.head())

print(df.info())

print(df.shape)

print (df.head())

fig = plt.figure(figsize=(20,3))

sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

df.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicidesper100kpop', 'country-year', 'HDI for year',

       'gdp_for_year', 'gdp_per_capita', 'generation']



df['gdp_for_year'] = df['gdp_for_year'].str.replace(',','').astype(int)

dfmod = df.drop(['HDI for year','country-year'],axis=1)
print(dfmod.head())

print(dfmod.info())

print(dfmod.shape)

print(dfmod.dtypes)

dfmod.describe()
dfmod.describe(include=['O'])
test=dfmod.groupby(["generation"])["suicides_no"].sum().reset_index().rename(columns={'generation':'generation', '':'suicide_no'})

#test=test.reset_index

#test.columns = ['generation', 'death_numbers']

test.head()

#np.shape(test)



sns.barplot(x="suicides_no", y="generation", data=test)

plt.show()


test2=dfmod.groupby(["sex"])["suicides_no"].sum().reset_index().rename(columns={'sex':'sex', '':'suicide_no'})



test2.head()

#np.shape(test)



sns.barplot(x="sex", y="suicides_no", data=test2)

plt.show()

test3=dfmod.groupby(["age"])["suicides_no"].sum().reset_index().rename(columns={'age':'age', '':'suicide_no'})

test3.head()

#np.shape(test)



sns.barplot(x="suicides_no", y="age", data=test3)

plt.show()
suicidebycountry = dfmod[['country','suicides_no']].groupby(['country']).sum()

suicidebycountry.plot(kind='bar', figsize=(40,10), fontsize=25)
suramerica = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay']

df_ne = df[df['country'].isin(suramerica)]

ax = df_ne.groupby(['country', 'year'])['suicides_no'].sum().unstack('country').plot(figsize=(10, 10))

ax.set_title('Número de suicidios en Ámerica del Sur', fontsize=20)

ax.legend(fontsize=15)

ax.set_xlabel('Year', fontsize=20)

ax.set_ylabel('suicides_no', fontsize=20, color='green')

plt.show()
#sns.pairplot(df)

#plt.show()

# https://python-graph-gallery.com/all-charts/
suramericarel = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay']

df_ne = df[df['country'].isin(suramericarel)]

ax = df_ne.groupby(['country', 'year'])['suicidesper100kpop'].sum().unstack('country').plot(figsize=(10, 10))

ax.set_title('Número de suicidios en Ámerica del Sur', fontsize=20)

ax.legend(fontsize=15)

ax.set_xlabel('Year', fontsize=20)

ax.set_ylabel('suicidesper100kpop', fontsize=20, color='green')

plt.show()