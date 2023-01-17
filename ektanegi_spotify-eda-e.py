# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
df = pd.read_csv("../input/spotifydata-19212020/data.csv")

print('No. of Rows',df.shape[0],'Number of Columns', df.shape[1])

print(df.columns)
df.head()
df.dtypes
categorical_features = [features for features in df if df[features].dtypes=='O']

categorical_features
for features in df[categorical_features]:

    print(features ,len(df[features].value_counts()))

    

# No. of values of each variable in Categorical features. 
continuous_features = [features for features in df if df[features].dtypes==('float64')]

continuous_features
discrete_features = [features for features in df if df[features].dtypes==('int64')]

discrete_features
for features in df[discrete_features]:

    print(features ,len(df[features].value_counts()))

    

# No. of values of each variable in Discrete features. 
fig , axes = plt.subplots(2,2,figsize=(15,10))

sns.countplot(df['mode'],ax=axes[0,0])

sns.countplot(df['explicit'],ax=axes[0,1])

sns.countplot(df['key'],ax=axes[1,0])

sns.distplot(df['popularity'],ax=axes[1,1],color='violet')

plt.show()
for features in df[categorical_features]:

    print(features , len(df[features].value_counts()))

    

# No. of Values of each variable in Categorical features
for features in df[continuous_features]:

    sns.distplot(df[features],color='green')

    plt.show()
#highly skewed features --> instrumentalness , liveness , loudness , speechiness
for features in df[continuous_features]:

    sns.boxplot(df[features],color='red')

    plt.show()
for features in df[continuous_features]:

    if(df[features]==0).any():

        print(features , len(df[df[features]==0][features]))

# outliers --> instrumentalness , liveness , loudness , speechiness , tempo
rel = df.copy()

variables = ['acousticness', 'danceability', 'energy','instrumentalness', 'key', 'liveness',

             'loudness', 'speechiness', 'tempo', 'valence']

year = range(2010,2021)



fig = plt.figure(figsize=(15,25))

for variable,num in zip(variables, range(1,len(variables)+1)):

    ax = fig.add_subplot(5,2,num)

    sns.scatterplot(variable, 'popularity', data=rel)

    plt.title('Relation between {} and Popularity'.format(variable))

    plt.xlabel(variable)

    plt.ylabel('Popularity')

fig.tight_layout(pad=0.5)
plt.figure(figsize=(12,6))

sns.set(style="whitegrid")

x = df.groupby("name")["popularity"].mean().sort_values(ascending=False).head(10)

ax = sns.barplot(x.index, x)

ax.set_title('Top Song with Popularity')

ax.set_ylabel('Popularity')

ax.set_xlabel('Songs')

plt.xticks(rotation = 90)

plt.figure(figsize=(12,6))

sns.set(style="whitegrid")

x = df.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(10)

ax = sns.barplot(x.index, x)

ax.set_title('Top Artists with Popularity')

ax.set_ylabel('Popularity')

ax.set_xlabel('Artists')

plt.xticks(rotation = 90)
plt.figure(figsize=(15, 8))

sns.set(style="whitegrid")

corr = df.corr()

sns.heatmap(corr,annot=True,cmap="YlGnBu")

plt.show
df[['artists','energy','acousticness']].groupby('artists').mean().sort_values(by='energy', ascending=False)[:10]

# i will not be using these columns as i infer these features dont play major role in predicting popularity of song



df.drop(['id','explicit','key','release_date','mode'], axis=1, inplace=True)