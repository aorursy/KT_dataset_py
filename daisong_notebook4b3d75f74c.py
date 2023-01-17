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
import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as ply

import statsmodels.formula.api as smf

from IPython import display

% matplotlib inline
data = pd.read_csv('../input/movie_metadata.csv')

data.sort_values('imdb_score', inplace=True, ascending=False)

data.drop_duplicates(subset=['director_name', 'movie_title', 'title_year'], inplace=True)
data.columns
data[data.num_voted_users >= 1000].head(50)[['director_name', 'movie_title', 'actor_1_name', 'actor_2_name', 'imdb_score' ,'movie_facebook_likes']]
director_stats = data.groupby('director_name')['imdb_score'].describe().reset_index()

director_stats_pivot = director_stats.pivot(index='director_name', columns='level_1', values='imdb_score')

director_stats_pivot = director_stats_pivot[director_stats_pivot['count'] >= 5]

director_stats_pivot.sort_values('mean', ascending=False, inplace=True)

director_stats_pivot.head(50)
top_director_names = director_stats_pivot.head(10).index
data[data.director_name.isin(top_director_names)].boxplot(column='imdb_score', by='director_name', figsize=[36, 12], fontsize=12)