# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings 

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df.isnull().sum()
df.info()
df['reviews_per_month'].head(10)
df['reviews_per_month'].mean()
df = df.dropna()
df.isnull().sum()
df.info()
df['price'].describe()
Q1 = df['price'].quantile(0.25)

Q3 = df['price'].quantile(0.75)

IQR=Q3-Q1

df = df[~((df['price']<(Q1-1.5*IQR))|(df['price']>(Q3+1.5-IQR)))]
df.info()
plt.figure(figsize=(10,7))

sns.distplot(df.price,color='r')

plt.xlabel("Price")

plt.title("Distribution of Price of Apartments")

plt.show()
import scipy.stats as st

st.shapiro(df.price)
df.room_type.unique()
pvt = df[df['room_type'] == 'Private room']

share = df[df['room_type'] == 'Shared room']

apt = df[df['room_type'] == 'Entire home/apt']
st.levene(pvt.price, share.price, apt.price)
plt.figure(figsize=(10,6))

sns.boxplot(y='price',x='room_type',data=df)

plt.show()
st.kruskal(pvt.price,share.price,apt.price)
ind = ['Private Rooms','Apartments','Shared Rooms']

x = pd.DataFrame([pvt.price.mean(),apt.price.mean(),share.price.mean()], index=ind)

x
x.plot.bar(color='g')

plt.title("Barplot of Mean Price Across Different Categories of Rooms")

plt.show()
st.f_oneway(pvt.price,share.price,apt.price)
x.plot.bar(color='r')

plt.title("Barplot of Mean Price Across Different Categories of Rooms")

plt.show()
df.neighbourhood_group.unique()
a = df[df['neighbourhood_group'] == 'Brooklyn']['price']

b = df[df['neighbourhood_group'] == 'Manhattan']['price']

c = df[df['neighbourhood_group'] == 'Queens']['price']

d = df[df['neighbourhood_group'] == 'Staten Island']['price']

e = df[df['neighbourhood_group'] == 'Bronx']['price']



st.kruskal(a,b,c,d,e)
st.f_oneway(a,b,c,d,e)
ind = ['Brooklyn','Manhattan','Queens','Staten Island','Bronx']

x = pd.DataFrame([a.mean(),b.mean(),c.mean(),d.mean(),e.mean()], index=ind)

x.plot.bar(color='m')

plt.show()
tab = pd.crosstab(df['room_type'],df['neighbourhood_group'])

st.chi2_contingency(tab)
ct = pd.crosstab(df['room_type'],df['neighbourhood_group'])

ct.plot.bar(stacked=True)

plt.show()
from statsmodels.formula.api import ols
model = ols("price~neighbourhood_group+host_name",data=df).fit()
# model.summary()
from statsmodels.stats.anova import anova_lm

anova_lm(model,typ=2)
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')

plt.show()
tab = pd.crosstab(df['neighbourhood'],df['neighbourhood_group'])

st.chi2_contingency(tab)