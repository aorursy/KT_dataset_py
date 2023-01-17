# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load data from url

url = "https://datafaculty.s3.us-east-2.amazonaws.com/Indore/song_football-class13.csv"



df = pd.read_csv(url, encoding="latin1")

df.head()
df.columns
# rename columns

df = df.rename(columns={'Player Id': 'pid', 'Tackles': 'tackles', 'Last_Name': 'lname', 'First_Name': 'fname'})

df.columns
df['name'] = df['fname'] + ' ' + df['lname']



df.drop(['fname', 'lname'], axis=1, inplace=True)
df.describe(include='all')
df.loc[(df.interception <= 10) & (df.tackles <= 10) & (df.duels <= 10)]
sns.boxplot(x=df['tackles'])
sns.boxplot(x=df['duels'])
# number of rows

len(df)
# NaN check

df.isna().sum()
# null check

df.isnull().sum()
# create the list of players

players = df['name'].values

players[:5]
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

numeric_cols
numeric_cols.remove('pid')

numeric_cols
feature_cols = ['tackles', 'duels', 'passes', 'interception']

ratio_cols = ['wontackles', 'wonduels', 'wonpasses',]
numeric_data = df[numeric_cols]

numeric_data.sample(7)
sns.set()

plt.legend(ncol=2, loc='upper right');
for col in feature_cols:

    plt.hist(df[col], alpha=0.5)
for col in feature_cols:

    sns.kdeplot(df[col], shade=True)
for col in ratio_cols:

    sns.kdeplot(df[col], shade=True)
for col in ratio_cols:

    plt.hist(df[col], alpha=0.5)
sns.kdeplot(numeric_data[feature_cols])
sns.kdeplot(numeric_data[ratio_cols])
with sns.axes_style('white'):

    sns.jointplot('wontackles', 'wonduels', df, kind='kde')
with sns.axes_style('white'):

    sns.jointplot('wontackles', 'wonpasses', df, kind='kde')
g = sns.PairGrid(numeric_data, vars=feature_cols, palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend()
sns.scatterplot(x='tackles', y='wontackles', hue='pid', palette="Set2", data=df)
from sklearn.cluster import KMeans
X = df[numeric_cols].values
kmeans = KMeans(n_clusters=4)

kmeans
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

y_kmeans
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# cluster columns

def cluster_cols(df, l, n):

    

    X = df[l].values

    kmeans = KMeans(n_clusters=n)



    kmeans.fit(X)

    

    y_kmeans = kmeans.predict(X)

    

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')



    centers = kmeans.cluster_centers_

    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    

    return y_kmeans

tackles = ['tackles', 'wontackles']



cluster_cols(df, tackles, 6)
df.pid.unique()[:5]
numeric_data.head()
numeric_data['cluster'] = cluster_cols(df, numeric_cols, 8)
numeric_data.head()
g = sns.PairGrid(numeric_data, vars=feature_cols, hue='cluster', palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend()
df['cluster'] = numeric_data['cluster']

df.head()
sns.scatterplot(x='tackles', y='wontackles', hue='cluster', palette="Set2", data=df)
df['cluster'].value_counts()
df['name'].loc[df.cluster == 3].unique()
df['name'].loc[df.cluster == 4].unique()
df.loc[df.cluster == 4]
df.loc[df.cluster == 7]