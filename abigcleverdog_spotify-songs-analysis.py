# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding = "ISO-8859-1")#, index_col='Unnamed: 0')

df.head()
cols = df.columns

cols
cols = list(map(lambda x: x.replace('.', '_'), cols))

cols
df.columns = cols

df.head()
df = df.rename(columns={'Loudness__dB__':'Loudness(dB)','Valence_': 'Valence', 'Length_':'Length', 'Acousticness__':'Acousticness', 'Speechiness_':'Speechiness'})

df.head()
df.info()
df.isnull().sum()
print(df.Artist_Name.nunique(), df.Genre.nunique())
f,ax = plt.subplots(2, 1, figsize=(8,10))

ax[0].set_title('Artists')

sns.countplot(y=df.Artist_Name, ax=ax[0], order = df.Artist_Name.value_counts().index)



ax[1].set_title('Genre')

sns.countplot(y=df.Genre, ax=ax[1], order = df.Genre.value_counts().index)
cols = df.columns.tolist()[4:]

cols
f, ax = plt.subplots((len(cols)+1)//2,2, figsize=(10, 20))

for i, col in enumerate(cols):

#     ax[i//2][i%2].set_title(col)

    sns.distplot(df[col], kde=False, ax=ax[i//2][i%2])
sns.regplot(x='Beats_Per_Minute', y='Popularity', data=df);
sns.jointplot(x='Beats_Per_Minute', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Energy', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Danceability', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Loudness(dB)', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Liveness', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Valence', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Length', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Acousticness', y='Popularity', kind='kde', data=df);
sns.jointplot(x='Speechiness', y='Popularity', kind='kde', data=df);
f, ax = plt.subplots(len(cols),1, figsize=(5, 50))

for i, col in enumerate(cols):

    sns.jointplot(x=col, y='Popularity', kind='kde', data=df, ax=ax[i]);
# Stolen from other kernel

from wordcloud import WordCloud

plt.style.use('seaborn')

wrds1 = df["Artist_Name"].str.split("(").str[0].value_counts().keys()



wc1 = WordCloud(scale=5,max_words=1000,colormap="rainbow",background_color="white").generate(" ".join(wrds1))

plt.figure(figsize=(12,18))

plt.imshow(wc1,interpolation="bilinear")

plt.axis("off")

plt.title("Artist Name with more songs in data ",color='b')

plt.show()