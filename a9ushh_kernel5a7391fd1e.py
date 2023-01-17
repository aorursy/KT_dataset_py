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
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("/kaggle/input/imdb-5000-movie-dataset/movie_metadata.csv")

df.head()
df.shape
df.info()
df.describe()
df['movie_imdb_link'].head()
## Ratings >7.5

df[df['imdb_score']>7.5].shape[0]
data_groupby = df.groupby(['imdb_score'])['movie_title'].count()

data_groupby.plot()

data_groupby = df.groupby(['duration'])['movie_title'].count()

data_groupby.plot()
df[df['duration'] <= 100].shape[0]

df[df['duration'] >= 180].shape[0]

df[df['language']=="English"].shape[0]
plt.figure(figsize=(10,10))

sns.countplot(x='language',data=df);

plt.xticks(rotation=90);
sns.countplot(x='color',data=df);

plt.figure(figsize=(10,10))

data_groupby = df.groupby(['title_year'])['gross'].count()

data_groupby.plot()

high_imdb = df.sort_values(by='imdb_score', ascending = False)

high_imdb=high_imdb.loc[:,['movie_title', 'imdb_score','title_year', 'language', 'country', 'budget', 'director_name', 'duration', 'gross' ]]

high_imdb.head(20)
hindi=high_imdb[high_imdb["language"]=="Hindi"]

hindi.head()
df.corr()
df['actor_1_name'].value_counts()[:10]

plt.figure(figsize=(20,20))

df['actor_1_name'].value_counts()[:100].plot(kind="bar")

plt.figure(figsize=(20,20))

df['actor_2_name'].value_counts()[:100].plot(kind="bar")

plt.figure(figsize=(20,20))

df['actor_3_name'].value_counts()[:100].plot(kind="bar")

plt.figure(figsize=(20,20))

df['country'].value_counts()[:100].plot(kind="bar")

df['title_year'].hist()