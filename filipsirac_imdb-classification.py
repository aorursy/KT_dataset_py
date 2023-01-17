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
imdb_movies = pd.read_csv("../input/imdbdataset/imdb-movies.csv")
df = pd.DataFrame(imdb_movies, columns=['genres', 'original_title'])

df.head()
#df.apply(pd.Series).set_index('original_title').genres.str.get_dummies(sep='|').stack().reset_index().drop(0, 1)#

#.merge(df, left_index = True, right_index = True).drop(["genres"], axis = 1) \

                           # .melt(id_vars = ['original_title'], value_name = "genres") \

                           # .drop("variable", axis = 1) \

                           #.dropna() \

                            

formated_movies = df.set_index('original_title').genres.str.get_dummies(sep='|').stack().reset_index().drop(0, 1)

formated_movies.to_csv('imdb_moives.txt'.format('__label__' + df.genres, df.original_title), sep='\t', index=False)
