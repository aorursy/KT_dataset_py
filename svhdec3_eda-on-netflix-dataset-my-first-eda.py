# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset_netflix = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
dataset_netflix.head()
dataset_netflix.shape
dataset_netflix.info()
dataset_netflix.isnull().sum()
dataset_netflix.type.unique()
np.sort(dataset_netflix.release_year.unique())
category = dataset_netflix.type.value_counts()

category
plt.pie(category,labels=["Movie", "TV Show"],autopct='%1.1f%%', explode=(0,0.1))

plt.title('Netflix Content Type')

plt.axis('equal')

plt.show()
count_by_year = dataset_netflix[dataset_netflix.release_year>=2000].groupby(["type","release_year"], as_index=False)["show_id"].count()

count_by_year.head()
plt.rcParams['figure.figsize'] = (20,10)

(count_by_year.pivot_table(index='release_year', columns='type', values='show_id',

                aggfunc='sum', fill_value=0)

   .plot.bar(stacked=False)

)

plt.show()
rating = dataset_netflix.groupby(["type", "rating"])["show_id"].count().reset_index(name='count').sort_values(by=["type","count"], ascending = False)

rating
sns.set(style="darkgrid")

g = sns.catplot(

    data=rating, kind="bar",

    x="rating", y="count", hue="type",

    ci="sd", palette="viridis", alpha=.6, height=6, aspect = 2

)

tvshow = dataset_netflix[dataset_netflix.type == "TV Show"]

tvshow['duration'] = tvshow['duration'].str.split(" ", n=1, expand = True)[0]

tvshow['duration'] = tvshow.duration.astype(int)

tvshow[tvshow.duration == tvshow.duration.max()][["title", "duration"]]
movie = dataset_netflix[dataset_netflix.type == "Movie"]

movie['duration'] = movie['duration'].str.split(" ", n=1, expand = True)[0]

movie['duration'] = movie.duration.astype(int)

movie[movie.duration.isin(movie['duration'].nlargest(3))][["title", "duration"]]
def split_dataframe_rows(df,column_selectors, row_delimiter):

    def _split_list_to_rows(row,row_accumulator,column_selector,row_delimiter):

        split_rows = {}

        max_split = 0

        for column_selector in column_selectors:

            split_row = str(row[column_selector]).split(row_delimiter)

            split_rows[column_selector] = split_row

            if len(split_row) > max_split:

                max_split = len(split_row)

            

        for i in range(max_split):

            new_row = row.to_dict()

            for column_selector in column_selectors:

                try:

                    new_row[column_selector] = split_rows[column_selector].pop(0)

                except IndexError:

                    new_row[column_selector] = ''

            row_accumulator.append(new_row)



    new_rows = []

    df.apply(_split_list_to_rows,axis=1,args = (new_rows,column_selectors,row_delimiter))

    new_df = pd.DataFrame(new_rows, columns=df.columns)

    return new_df



#Reference: https://gist.github.com/jlln/338b4b0b55bd6984f883 - gnespatel1618 commented on Sep 20, 2018
split_country = split_dataframe_rows(movie, ['country'], ',')

split_country['country'] = split_country['country'].str.lstrip()

split_country['country'] = split_country['country'].str.rstrip()

summary = split_country[(split_country['country'].isin(['United States', 'India', 'United Kingdom', 'Canada', 'Spain', 'Mexico'])) & (split_country['release_year']>=2000)].groupby(['country', 'release_year'])['show_id'].count().reset_index(name='count').sort_values(by=["count"], ascending = False)



pic = summary.pivot("release_year", "country", "count")

pic

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(pic, annot=False, linewidths=.5, ax=ax)

plt.show()

split_cast = split_dataframe_rows(dataset_netflix, ['cast'], ',')

split_cast['cast'] = split_cast['cast'].str.lstrip()

split_cast['cast'] = split_cast['cast'].str.rstrip()

#x = split_cast.groupby(['cast', 'type'])['show_id'].count().reset_index(name='count').sort_values(by=["cast"], ascending = False)

#x.sort_values(by = ['cast'], ascending = True)

CastedinMovieandTVShow = pd.DataFrame(split_cast.groupby(['cast'])['type'].nunique()).reset_index()

pd.set_option("max_rows", None)

CastedinMovieandTVShow[CastedinMovieandTVShow.type >= 2]['cast'].head(100)
