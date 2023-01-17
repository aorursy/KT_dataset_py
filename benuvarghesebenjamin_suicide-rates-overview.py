import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
data.head()
data.columns
data.rename(columns={'suicides/100k pop':'Suicides Per 100k Pop', ' gdp_for_year ($) ':'GDP For Year',

                          'gdp_per_capita ($)':'GDP Per Capita'}, inplace=True)
data.shape
data.isnull().sum()
data.drop(['HDI for year'],axis=1,inplace=True)
data.head()
data.hist(grid=True,figsize=(15,20),color='lime')

plt.show()
alpha = 0.7

plt.figure(figsize=(10,25))

sns.countplot(y='country', data=data, alpha=alpha)

plt.title('Data by country')

plt.show()
plt.rcParams['figure.figsize']=(6,6)
data['sex'].value_counts().plot.bar(color='red')
sns.barplot(x='sex', y='suicides_no', hue='age', data=data)
sns.barplot(x='sex', y='suicides_no', hue='generation', data=data)
plt.figure(figsize=(30,10))

y = data['year']

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y,label='count')
plt.figure(figsize=(14,4))

plt.subplot(121)

plt.title('Suicide Number')

sns.distplot(data['suicides_no'], hist=False)

plt.subplot(122)

plt.title('Suicide Number Per 100k population')

sns.distplot(data['Suicides Per 100k Pop'], hist=False)

plt.tight_layout()
data['GDP For Year'] = data['GDP For Year'].apply(lambda x: x.replace(',','')).astype(float)
plt.subplots(figsize=(12,6))

sns.heatmap(data.corr(), annot=True, linewidths = 0.3)

plt.title('Correlation of the dataset', size=16)

plt.show()
data.groupby(["sex"])["suicides_no"].sum().reset_index()
sns.barplot(x="sex", y="suicides_no", data=data, palette="Blues_d")

plt.show()
country= data.groupby(['country'])['suicides_no'].sum().reset_index()

country = country.sort_values('suicides_no',ascending=False)

country=country.head()

plt.subplots(figsize=(15,6))

sns.barplot(x='country', y='suicides_no', data=country,color='fuchsia')