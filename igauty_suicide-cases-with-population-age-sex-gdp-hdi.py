# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import datetime

import time

import warnings

warnings.filterwarnings('ignore')



from scipy.stats import norm

from scipy import stats

from sklearn import preprocessing



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

sns.set_style('whitegrid')

%matplotlib inline
#data = pd.read_csv('/media/vishwadeepg/New Volume/Work/0. Gauty/Kernal/suicide_rates_overview/master.csv')

data = pd.read_csv('../input/master.csv')
#Size of datasets

print("Size of dataset  (Rows, Columns): ",data.shape)
# data snapshot

data.head()
data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_avg', 'country_year', 

                'HDI_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation']
#general information of data

print("Data info: ",data.info())
print("Datatypes of dataset are:")

print(data.dtypes.value_counts())
data.describe()
def missing_check(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    #print("Missing check:",missing_data )

    return missing_data

missing_check(data)
descending_order = data['country'].value_counts().sort_values(ascending=True).index

figure = plt.figure(figsize=(15,30))

ax = sns.countplot(y=data['country'], data=data, order=descending_order)
ax = sns.countplot(x=data['sex'], data=data)
descending_order = data['year'].value_counts().sort_values(ascending=True).index

figure = plt.figure(figsize=(10,15))

ax = sns.countplot(y=data['year'], data=data, order=descending_order)
figure = plt.figure(figsize=(10,8))

ax = sns.countplot(y=data['age'], data=data)
data.population.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)
data.suicides_avg.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)
data.gdp_per_capita.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)
#data.gdp_for_year.hist(figsize=[10,5], xlabelsize=10, ylabelsize=10)
data.HDI_for_year = data.HDI_for_year.fillna(0)

data.HDI_for_year.hist(figsize=[10,5],bins=50, xlabelsize=10, ylabelsize=10)
figure = plt.figure(figsize=(10,8))

ax = sns.countplot(y=data['generation'], data=data)
figure = plt.figure(figsize=(15,8))

data.groupby(by=['country'])['suicides_no'].sum().reset_index().sort_values(['suicides_no'],

                    ascending=True).tail(30).plot(x='country',y='suicides_no',kind='bar', figsize=(15,8))
figure = plt.figure(figsize=(10,15))

data.groupby(by=['country'])['suicides_no'].sum().reset_index().sort_values(['suicides_no'],

                    ascending=True).head(30).plot(x='country',y='suicides_no',kind='bar', figsize=(15,8))
figure = plt.figure(figsize=(15,8))

plt.scatter(data.year, data.suicides_no, color='g')

plt.xlabel('year')

plt.ylabel('Number of suicides')

plt.show()
plt.figure(figsize=(15,10))

ax = sns.barplot(x='age', y='suicides_no', data=data);
plt.figure(figsize=(8,10))

ax = sns.barplot(x="sex", y="suicides_no", data=data)
figure = plt.figure(figsize=(20,10))

ax = sns.regplot(x=data['population'],y='suicides_no', data=data, color='m')
figure = plt.figure(figsize=(20,10))

ax = sns.regplot(x=data['suicides_avg'],y='suicides_no', data=data, color='r')
"""

figure = plt.figure(figsize=(20,10))

plt.scatter(data.gdp_for_year, data.suicides_no, color='r')

plt.xlabel('suicides_avg')

plt.ylabel('Number of suicides')

plt.show()

"""
figure = plt.figure(figsize=(20,10))

ax = sns.regplot(x=data['gdp_per_capita'],y='suicides_no', data=data)
df = data[data['HDI_for_year']>0]

figure = plt.figure(figsize=(20,10))

ax = sns.regplot(x=df['HDI_for_year'],y='suicides_no', data=df)
figure = plt.figure(figsize=(20,10))

data_scaled = data.loc[:,['gdp_per_capita','suicides_avg']]

data_scaled = (data_scaled - data_scaled.mean()) / data_scaled.std()

sns.scatterplot(data=data_scaled,x='gdp_per_capita',y='suicides_avg', color='b')
figure = plt.figure(figsize=(20,10))

data_scaled = df.loc[:,['HDI_for_year','suicides_avg']]

data_scaled = (data_scaled - data_scaled.mean()) / data_scaled.std()

sns.scatterplot(data=data_scaled,x='HDI_for_year',y='suicides_avg',color='g')
plt.figure(figsize=(15,10))

ax = sns.barplot(x='generation', y='suicides_no', data=data);
plt.figure(figsize=(20,10))

sns.barplot(data=data,x='sex',y='suicides_no',hue='age')
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']<2004],x='sex',y='suicides_no',hue='year')
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']>2004],x='sex',y='suicides_no',hue='year')
plt.figure(figsize=(20,10))

ax = sns.barplot(x="generation", y="suicides_no", hue="sex", data=data)
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']<2004],x='year',y='suicides_no',hue='age')
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']>2003],x='year',y='suicides_no',hue='age')
plt.figure(figsize=(20,10))

ax = sns.barplot(x="generation", y="suicides_no", hue="age", data=data)
data.head()
le = preprocessing.LabelEncoder()

data['sex'] = le.fit_transform(data['sex'])

data['generation'] = le.fit_transform(data['generation'])

data['age'] = le.fit_transform(data['age'])

data['country'] = le.fit_transform(data['country'])

df1 = data[['country', 'sex', 'age',  'population','suicides_avg', 'HDI_for_year', 'gdp_for_year',

            'gdp_per_capita', 'generation', 'suicides_no']]
data_corr = df1.corr()['suicides_no'][:-1] # -1 because the latest row is Target

golden_features_list = data_corr.sort_values(ascending=False)

golden_features_list
corr = df1.corr() 

plt.figure(figsize=(12, 10))



sns.heatmap(corr, 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);