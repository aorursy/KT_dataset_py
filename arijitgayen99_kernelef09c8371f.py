import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option("display.max_columns", 100)

import os

print(os.listdir("../input"))

df = pd.read_csv("../input/master.csv")

df.head()
df.describe()
df1 = df.drop(['HDI for year', 'country-year'], axis=1)

df1 = df1.rename(columns={'suicides/100k pop': 'suicide/100k', ' gdp_for_year ($)': 'GDP', 'gdp_per_capita ($)' : 'GDP/capita'})



df1.head()
plt.figure(figsize=(14,10))

sns.barplot(x='age',y='suicides_no',data=df1.groupby(['age','sex']).sum().reset_index(),hue='sex')
plt.figure(figsize=(14,10))

sns.scatterplot(df1["GDP/capita"], df1["suicides_no"], hue = df1["sex"])
plt.figure(figsize = (14, 10))

sns.barplot(x='generation',y='suicides_no',data=df1.groupby(['generation','sex']).sum().reset_index(),hue='sex')
plt.figure(figsize = (20, 10))

sns.barplot(x='year',y='suicides_no',data=df1.groupby(['year','sex']).sum().reset_index(),hue='sex')

plt.figure(figsize = (10,20))

sns.countplot(y = 'country', data = df1, alpha = 0.5)

for i in range(50,401,50):

    plt.axvline(i)
plt.figure(figsize = (12,10))



plt.subplot(2, 1, 1)

sns.lineplot(x = 'year', y = 'suicide/100k', data = df1)

plt.axhline(df1['suicide/100k'].mean())

plt.grid(True)



plt.subplot(2, 1, 2)

sns.lineplot(x = 'year', y = 'suicide/100k', data = df1, hue = 'sex')

plt.grid(True)

plt.axhline(np.mean(list(df1[df1.sex == 'female'].iloc[:,6])), color = 'coral')

plt.axhline(np.mean(list(df1[df1.sex == 'male'].iloc[:,6])))
plt.figure(figsize = (16,6))



sns.countplot(x = 'year', data = df1, alpha = 0.5)