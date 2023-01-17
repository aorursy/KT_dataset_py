# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/winemag-data_first150k.csv")
# Let us look at some sample data
df.head()
df.shape
categorical_features = df.select_dtypes(include=[np.object])
categorical_features.columns

numeric_features = df.select_dtypes(include=[np.number])
numeric_features.columns
#check the decoration
df.columns

df['price'].describe()
print("Skewness: %f" % df['price'].skew())
print("Kurtosis: %f" % df['price'].kurt())
df['country'].value_counts().head(10).plot.bar()
(df['country'].value_counts().head(10)/len(df)).plot.bar()
(df['province'].value_counts().head(10) / len(df)).plot.bar()
df['points'].value_counts().sort_index().plot.bar()
df.plot.scatter(x='points', y='price')
df.plot.hexbin(x='price', y='points')
df[df['price'] < 200].plot.hexbin(x='price', y='points', gridsize=15)
ax = df['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title("Rankings Given by Wine Magazine", fontsize=20)
#subplot

fig, axarr = plt.subplots(2, 1, figsize=(16, 12))

df['points'].value_counts().sort_index().plot.bar(
    ax=axarr[0]
)

df['province'].value_counts().head(20).plot.bar(
    ax=axarr[1]
)
#SEABORN
sns.distplot(df['points'])
sns.jointplot(x="points", y="price", data = df)
sns.boxplot(x="points", y="price", data = df)
#(sns.boxplot(x="variety", y="points", data = df).head(5))
var = 'points'
data = pd.concat([df['price'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=var, y="price", data=data)
fig.axis(ymin=0, ymax=2000);
df1= df[df.variety.isin(df.variety.value_counts().head(5).index)]

sns.boxplot(
    x = 'variety',
    y = 'points',
    data = df1, 
    
)
df =df.drop(['Unnamed: 0'], axis=1)
#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corrmat, vmax=.8, square=True);
#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);

sns.set()
columns = ['price', 'points']
sns.pairplot(df[columns],size = 10 ,kind ='scatter',diag_kind='kde')
plt.show()