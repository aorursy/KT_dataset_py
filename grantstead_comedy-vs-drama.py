# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv("../input/movie_metadata.csv")
movies.head()
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
some_movies = movies[movies.title_year >= 2000].dropna()

comedies = some_movies[some_movies.genres.str.contains("Comedy")]

dramas = some_movies[some_movies.genres.str.contains("Drama")]
chart_comedies = sns.kdeplot(comedies.title_year, comedies.imdb_score, cmap="Blues", shade=False, shade_lowest=False)

chart_dramas = sns.kdeplot(dramas.title_year, dramas.imdb_score, cmap="Reds", shade=False, shade_lowest=False)

import seaborn as sns

fig, ax = plt.subplots(figsize=(12,6))

p = sns.violinplot(data=movies, x='content_rating', y='imdb_score', ax=ax)