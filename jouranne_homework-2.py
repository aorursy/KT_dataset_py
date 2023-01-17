import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/master.csv")



# Print the head of df

print(df.head())



# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)
df.iloc[:,~df.columns.isin(['country'])].describe()
df.fillna(df.mean())

df.iloc[:,~df.columns.isin(['country'])].describe()

dh=df[((df.year == 2000) ) & (df.sex != 'Both')]

print(dh.iloc[:,~dh.columns.isin(['country'])].describe())



dg=df[((df.year == 1999) ) & (df.sex != 'Both')]

print(dg.iloc[:,~dg.columns.isin(['country'])].describe())

dg=df[((df.year == 1999) ) & (df.sex != 'Both')]

dg.iloc[:,~dg.columns.isin(['country'])].describe()
dg=df[((df.year == 1990)  ) & (df.sex != 'Both')]

dg.iloc[:,~dg.columns.isin(['country'])].describe()
years=[1990,2000]

dp=df[df.year.isin(years)]

plt.figure(figsize=(20,20))

sns.barplot(x="suicides_no",y="country", hue="year", data = dp)

plt.xlabel("Country",fontsize=15)

plt.ylabel("Suicides",fontsize=15)

plt.title("1990 vs 2000",fontsize=15)

plt.show()
f, axes = plt.subplots(1,3, figsize=(20, 4))

sns.distplot( dp["suicides_no"], ax=axes[0])

sns.distplot( dp["population"], ax=axes[1])

sns.distplot( dp["gdp_per_capita ($)"], ax=axes[2])


corr=dp.iloc[:,~dp.columns.isin(['generation','country-year','year'])].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})