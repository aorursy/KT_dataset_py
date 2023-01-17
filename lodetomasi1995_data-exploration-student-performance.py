import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('../input/StudentsPerformance.csv')

# Any results you write to the current directory are saved as output.
df.head(10) #summary

df.columns #names of columns
plt.figure(figsize=(12,6))

df.gender.value_counts()

sns.countplot(x="gender", data=df, palette="bwr")

plt.show()
df.gender.value_counts()

plt.figure(figsize=(12,6))

sns.countplot(x="parental level of education", data=df, palette="bwr")

plt.show()
pd.crosstab(df.gender,df['test preparation course']).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Completed Pre-test divided by gender')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Completed", "Not Completed"])

plt.ylabel('Frequency')

plt.show()
df['test preparation course'].value_counts()

plt.figure(figsize=(12,6))



sns.countplot(x="test preparation course", data=df, palette="bwr")

plt.show()
plt.figure(figsize=(8,6))

df['race/ethnicity'].value_counts().head(10).plot.bar()

df_treat = pd.get_dummies(df)

df_treat.dtypes.value_counts()

df_treat.head(10)
fig = plt.figure(figsize=(8,10))

ax1 = fig.add_subplot(311)

ax2 = fig.add_subplot(312)

ax3 = fig.add_subplot(313)

sns.distplot(df['math score'],ax=ax1,color="y")

sns.distplot(df['writing score'],ax=ax2,color="r")

sns.distplot(df['reading score'],ax=ax3)

g = sns.FacetGrid(df, hue='gender', size = 7)

g.map(plt.scatter, 'math score','reading score', edgecolor="w")

g.add_legend()
g = sns.FacetGrid(df, hue='test preparation course', size = 7)

g.map(plt.scatter, 'math score','reading score', edgecolor="w")

g.add_legend()
p = sns.FacetGrid(data = df, hue = 'gender', size = 5, legend_out=True)

p = p.map(sns.kdeplot, 'math score')

plt.legend()

p = sns.FacetGrid(data = df, hue = 'gender', size = 5, legend_out=True)

p = p.map(sns.kdeplot, 'reading score')

plt.legend()

p = sns.FacetGrid(data = df, hue = 'gender', size = 5, legend_out=True)

p = p.map(sns.kdeplot, 'writing score')



plt.legend()