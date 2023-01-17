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
import matplotlib.pyplot as plt
import seaborn as sns

anime_data = pd.read_csv("/kaggle/input/anime-recommendations-database/anime.csv", index_col="anime_id")
rating_data = pd.read_csv("/kaggle/input/anime-recommendations-database/rating.csv")
anime_data.head()
sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sorted_by_rating = anime_data.sort_values(by=["rating"], ascending=False)
sns.barplot(x=sorted_by_rating.name[:5], y=sorted_by_rating.rating[:5])
final_rating = rating_data.groupby('anime_id').sum()
anime_data_rated = anime_data.join(final_rating, lsuffix="_gen", rsuffix="_user")
anime_data_rated.sort_values(by=['rating_user'], ascending=False, inplace=True)
plt.figure(figsize=(14,8))
plt.title("Top 5 depending of users")
sns.barplot(x=anime_data_rated.name[:5], y=anime_data_rated.rating_user[:5])
#anime_data.rating = anime_data.rating.fillna(anime_data.rating.mean())
#anime_data.rating.isnull().sum()
sns.set_style("darkgrid")
fig = plt.figure(figsize=(12,8))
plt.title("Correlation between members and rating")
sns.scatterplot(x=anime_data.rating, y=anime_data.members, hue=anime_data.type)
anime_data.corr()
plt.figure(figsize=(14,8))
sns.set_style("darkgrid")
sns.regplot(x=anime_data_rated.rating_gen, y=anime_data_rated.rating_user)
anime_data_rated[['rating_gen','rating_user']].corr()
sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sorted_by_rating = anime_data.sort_values(by=["rating"])
sns.barplot(x=sorted_by_rating.name[:5], y=sorted_by_rating.rating[:5])
worst_anime = anime_data_rated.sort_values(by=['rating_user'])
plt.figure(figsize=(14,8))
sns.barplot(x=worst_anime.name[:5], y=worst_anime.rating_user[:5])
anime_types = list(anime_data.type.unique())
anime_types.pop()
anime_types
anime_data.type.value_counts()
anime_dic = {}
for t in anime_types:
    serie_type = anime_data.loc[anime_data.type == t].rating.sort_values(ascending=False).dropna()
    anime_dic[t] = serie_type[:100].tolist()
anime_types_data = pd.DataFrame(anime_dic)

plt.figure(figsize=(14,8))
sns.heatmap(data=anime_types_data)
plt.ylabel("Anime")
anime_dic_u = {}
for t in anime_types:
    serie_type = anime_data_rated.loc[anime_data_rated.type == t].rating_user.sort_values(ascending=False).dropna()
    anime_dic_u[t] = serie_type[:100].tolist()
anime_types_u_data = pd.DataFrame(anime_dic_u)

plt.figure(figsize=(14,8))
sns.heatmap(data=anime_types_u_data)
plt.ylabel("Anime")
plt.figure(figsize=(14,8))
sns.distplot(a=anime_data_rated.rating_gen, kde=False)