import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

%matplotlib inline
train_df = pd.read_csv('../input/falldeteciton.csv')
train_df.head()
train_df.info()
train_df.describe()
# check for missing data

print(train_df.isnull().sum())
sns.distplot(train_df['ACTIVITY'], kde=True)
sns.distplot(train_df['TIME'], kde=True)
sns.distplot(train_df['SL'], kde=True)
sns.distplot(train_df['EEG'], kde=True)
sns.distplot(train_df['HR'], kde=True)
sns.distplot(train_df['CIRCLUATION'], kde=True)
plt.rcParams["figure.figsize"] = (18,9)

plt.rc('xtick', labelsize=20) 

plt.rc('ytick', labelsize=20)



sns.heatmap(train_df)
g = sns.FacetGrid(train_df, col="ACTIVITY") 

g.map(plt.scatter, "TIME", "SL")
g = sns.FacetGrid(train_df, col="ACTIVITY") 

g.map(plt.scatter, "TIME", "CIRCLUATION")
plt.rcParams["figure.figsize"] = (18,9)

plt.rc('xtick', labelsize=20) 

plt.rc('ytick', labelsize=20)

scatter_matrix(train_df)

plt.show()