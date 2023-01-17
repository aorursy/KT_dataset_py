import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt;

import seaborn as sns

import os

os.chdir("../input")

df=pd.read_csv('master.csv')

df.describe()
p=df.groupby(['age']).sum().reset_index('age')

temp_df=p.filter(['age', 'suicides_no'], axis=1)

sns.barplot(x="age", y="suicides_no", data=temp_df)
plt.pie(p.suicides_no, labels=p.age); 

plt.show()
p=(df.groupby(['age'])['suicides_no'].sum() / df.groupby(['age'])['population'].sum()).reset_index('age').sort_values(0)
sns.barplot(x='age', y=0, data=p)
coorelation=df.corr();

sns.heatmap(coorelation, xticklabels=coorelation.columns.values, yticklabels=coorelation.columns.values)
p=df.groupby(['gdp_per_capita ($)']).sum().reset_index('gdp_per_capita ($)')

p=p.filter(['gdp_per_capita ($)', 'suicides_no'], axis=1)

sns.barplot(x='gdp_per_capita ($)', y="suicides_no", data=p)

p=df.groupby(['HDI for year']).sum().reset_index('HDI for year')

p=p.filter(['HDI for year', 'suicides_no'], axis=1)

sns.barplot(x='HDI for year', y="suicides_no", data=p)