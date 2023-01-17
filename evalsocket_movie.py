# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra}

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv('../input/movie_metadata.csv')



data.columns.values
data[data.director_name == 'James Cameron'][['movie_title', 'budget', 'gross']]
a = data[['movie_title', 'budget', 'gross']]

a = a.set_index('movie_title')

profit = a['gross'] / a['budget']

a['profit_ratio'] = profit

a.sort('profit_ratio', ascending=False)['profit_ratio'].head(10).plot('bar')
directors = data.groupby('director_name').sum()

movie_count = data.groupby('director_name').count()['color']

directors['movie_count'] = movie_count

#directors = data.groupby('director_name').sum().sort('budget', ascending=False)[['budget']]



avg_budget = directors['budget'] / directors['movie_count']

directors['avg_budget'] = avg_budget

directors.sort('avg_budget', ascending=False)[['budget', 'movie_count', 'avg_budget']]
