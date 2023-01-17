# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/movie_metadata.csv')
dirs = df.groupby('director_name').count()

directors = dirs[dirs.title_year>=5].index.tolist()
g = df[['director_name','imdb_score','title_year']]

g = g[g.director_name.isin(directors)]
ratings = g.groupby(['director_name','title_year']).imdb_score.mean().reset_index()



ratings['title_year'] = ratings.title_year.astype(int)
g = sns.FacetGrid(ratings, col="director_name",col_wrap=5)

g = g.map(plt.plot, "title_year", "imdb_score", marker = 'o')



g.set_xticklabels(rotation = 45)