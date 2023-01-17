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
netflix_movies_file = '../input/netflix-shows/netflix_titles.csv'
netflix_movies = pd.read_csv(netflix_movies_file)
netflix_movies.head()
netflix_movies.dtypes
netflix_mdata = netflix_movies.dropna()
netflix_mdata.isnull().sum()
netflix_mdata.describe()
sns.boxplot(x="type", y="release_year", data=netflix_mdata)
netflix_mdata['type'].value_counts().to_frame()
movies_filter = netflix_mdata['type'] == 'Movie'
tvshow_filter = netflix_mdata['type'] == 'TV Show'
movies_netflix = netflix_mdata[movies_filter]
tvshows_netflix = netflix_mdata[tvshow_filter]
movies_netflix.head()
from collections import Counter
genre_1 = movies_netflix['listed_in']
genre_count = pd.Series(dict(Counter(','.join(genre_1).replace(' ,',',').replace(', ',',')
                                       .split(',')))).sort_values(ascending=False)

genre_count
geners = list(genre_count.keys())
numbers = list(genre_count)
plt.barh(geners,numbers,color='Purple')
plt.ylabel('Tags of netflix')
plt.xlabel('Number of films w/ the tag')
plt.title('Number of films per gender netflix')
plt.figure(figsize=(20,12))



genre_1 = tvshows_netflix['']
genre_count = pd.Series(dict(Counter(','.join(genre_1).replace(' ,',',').replace(', ',',')
                                       .split(',')))).sort_values(ascending=False)

genre_count