# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
main_df = pd.read_csv("/kaggle/input/steam-store-games/steam.csv", parse_dates = ["release_date"])

description_df = pd.read_csv("/kaggle/input/steam-store-games/steam_description_data.csv")

df = main_df.join(description_df)

df = df[["appid", "name", "release_date", "platforms", "genres", "short_description", "categories", "owners"]]
# Preprocess owners column to show the avg instead of an lower and upper bound

df['owners'] = df['owners'].str.split('-').apply(lambda x: (int(x[0]) + int(x[1])) / 2).astype(int)

df = df.rename(columns={'owners': 'avg_owners'})



df["genres"] = df["genres"].str.split(";")

mlb = MultiLabelBinarizer(sparse_output=True)

df = df.join(

            pd.DataFrame.sparse.from_spmatrix(

                mlb.fit_transform(df.pop('genres')),

                index=df.index,

                columns=mlb.classes_))

genres = mlb.classes_

df["platforms"] = df["platforms"].str.split(";")

mlb = MultiLabelBinarizer(sparse_output=True)

df = df.join(

            pd.DataFrame.sparse.from_spmatrix(

                mlb.fit_transform(df.pop('platforms')),

                index=df.index,

                columns=mlb.classes_))

print(platforms)

platforms = mlb.classes_

df["categories"] = df["categories"].str.split(";")

mlb = MultiLabelBinarizer(sparse_output=True)

df = df.join(

            pd.DataFrame.sparse.from_spmatrix(

                mlb.fit_transform(df.pop('categories')),

                index=df.index,

                columns=mlb.classes_))

categories = mlb.classes_

print(categories)

df.head()
genre_cols = df.columns[5:34] # Update index in case of adding or removing columns to the dataset



total_owners_per_genre = df[genre_cols].multiply(df['avg_owners'], axis='index').sum()

average_owners_per_genre = total_owners_per_genre / df[genre_cols].sum()



fig, ax1 = plt.subplots()



color = 'tab:blue'

average_owners_per_genre.sort_index(ascending=False).plot.barh(ax=ax1, color=color, alpha=.5, position=.2, label='1')

ax1.set_xlabel('average owners per genre', color=color)

ax1.tick_params(axis='x', labelcolor=color)



plt.tight_layout()

plt.show()
df1 = df[['release_date', 'VR Support']].copy()

df1['VR Support'] = df1['VR Support'].sparse.to_dense()

df1 = df1.groupby(df['release_date'].dt.year)['VR Support'].sum().reset_index()

fig, ax = plt.subplots()

ax.plot(df1['release_date'], df1['VR Support'], linestyle ='solid')

plt.title('Number of games that support VR over time')

ax.set_xlabel(xlabel = 'Year')

ax.set_ylabel(ylabel = 'Count')

plt.show()
top_categories = df[categories].sum().sort_values(ascending=False).index[:10]

top_genres = df[genres].sum().sort_values(ascending=False).index[:10]

corr = df.corr()[top_categories].transpose()[top_genres]

plot = sns.heatmap(corr)

plot.set_title('Correlation between game genres and categories.')
df1 = df[['release_date', 'linux', 'mac', 'windows']].copy()

df1['linux'] = df1['linux'].sparse.to_dense()

df1['mac'] = df1['mac'].sparse.to_dense()

df1['windows'] = df1['windows'].sparse.to_dense()



df1 = df1.groupby(df['release_date'].dt.year).sum().reset_index()

fig, ax = plt.subplots()

ax.plot(df1['release_date'], df1[['linux', 'mac', 'windows']], linestyle ='solid')

plt.title('Number of games per platform over time')

ax.set_xlabel(xlabel = 'Year')

ax.legend(['linux', 'mac', 'windows'])

ax.set_ylabel(ylabel = 'Count')

plt.show()