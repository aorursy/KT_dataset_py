# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from IPython.display import display



%matplotlib inline
data = pd.read_csv('/kaggle/input/imdb-5000-movie-dataset/movie_metadata.csv')

df = data.copy()



df
display(df.describe(exclude='object')); display(df.describe(include='object'))
df.info()
df.isna().sum()
cat_cols = list(df.select_dtypes(include='object').columns)

num_cols = list(df.select_dtypes(exclude='object').columns)



print(f'Categorical columns: {cat_cols}')

print(f'Continuous columns: {num_cols}')
df[cat_cols] = df[cat_cols].astype('category')
_, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(df[num_cols].corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True, ax=ax)
sns.pairplot(df[num_cols])
_, ax = plt.subplots(figsize=(18, 6))

sns.regplot(x=df['budget'].apply(np.log10), y=df['gross'], ax=ax)
df['director_facebook_likes'].plot.hist(bins=20, log=True)
_, ax = plt.subplots(figsize=(24, 6))

sns.barplot(x='country', y='gross', data=df, ax=ax, color='#30BA8F')

_ = plt.xticks(rotation=90)
_, ax = plt.subplots(figsize=(10, 6))

sns.countplot(x=df['imdb_score'].apply(np.round).astype('category'), 

              ax=ax)
_, ax = plt.subplots(figsize=(14, 6))

sns.barplot(x=df['imdb_score'].apply(np.round), y=df['gross'], ax=ax)



plt.xlabel('IMDB Score'); plt.xlabel('Year');
_, ax = plt.subplots(figsize=(24, 8))

sns.countplot(x=df['title_year'].astype('category'), ax=ax)

plt.xlabel('Year')

_ = plt.xticks(rotation=90)
gb_mean = df.groupby(['title_year'])['gross'].sum()



plt.figure(figsize=(10, 4))

gb_mean.iloc[-30:].plot()

plt.ylabel('Total Box-Office Collection')

plt.xlabel('Year')

_ = plt.xticks(rotation=90)
gb_mean = df.groupby(['director_name'])['gross'].max().dropna().sort_values(ascending=True).tail(10)



plt.figure(figsize=(6, 4))

gb_mean.plot.barh()

plt.xlabel('Gross Box-Office Collection')

_ = plt.xticks(rotation=90)
gb_mean = df.groupby(['director_name'])['imdb_score'].max().dropna().sort_values(ascending=True).tail(10)



plt.figure(figsize=(6, 4))

gb_mean.plot.barh()

plt.xlim(right=10)

plt.xlabel('IMDB Score')

_ = plt.xticks(rotation=90)
gb_mean = df.groupby(['actor_1_name'])['gross'].max().dropna().sort_values(ascending=True).tail(10)



plt.figure(figsize=(6, 4))

gb_mean.plot.barh()

plt.xlabel('Gross Box-Office Collection')

_ = plt.xticks(rotation=90)
gb_mean = df.groupby(['actor_1_name'])['imdb_score'].max().dropna().sort_values(ascending=True).tail(10)



plt.figure(figsize=(6, 4))

gb_mean.plot.barh()

plt.xlim(right=10)

plt.xlabel('IMDB Score')

_ = plt.xticks(rotation=90)
from collections import defaultdict

d = defaultdict(list)



tmp = df['genres'].str.split('|')

tmp_dict = tmp.to_dict()



for idx, genres in tmp_dict.items():

    for g in genres:

        d[g].append(idx)
total_gross_per_genre = pd.DataFrame([[genre, df.loc[d[genre]].gross.sum()] for genre in d.keys()], columns=['genre', 'gross_mean'])

total_gross_per_genre = total_gross_per_genre.dropna().sort_values(by='gross_mean').reset_index(drop=True)

# total_gross_per_genre
total_gross_per_genre.plot.barh(x='genre', y='gross_mean', figsize=(14, 8), color=['gray'], legend=False)

plt.xlabel('Total Box-Office Collection');