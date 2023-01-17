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
data = pd.read_csv('/kaggle/input/golden-globe-awards/golden_globe_awards.csv')
data.head()
(data['year_award'] - data['year_film']).median()
yearly_data = data.loc[(data['year_award'] >= 2000) & (data['year_award'] <= 2020) & (data['win'] == True)]
import seaborn as sns

ax = sns.barplot(x="year_award", y="category", data=yearly_data)
ax = sns.countplot(y = "category", data=yearly_data)
first_10_actor = yearly_data.groupby('nominee').count().sort_values('film',ascending = False).head(10)
first_10_actor
ax = sns.barplot(x="film", y=first_10_actor.index, data=first_10_actor)

ax.set(xlabel='count of artist award', ylabel='artist names')
first_10_film = yearly_data.groupby('film').count().sort_values('nominee',ascending = False).head(10)
first_10_film
ax = sns.barplot(x="nominee", y=first_10_film.index, data=first_10_film)

ax.set(xlabel='count of film award', ylabel='film names')
all_time = data.loc[ (data['win'] == True)]
first_10_best_artist = all_time.groupby('nominee').count().sort_values('film',ascending = False).head(10)
first_10_best_artist
ax = sns.barplot(x="film", y=first_10_best_artist.index, data=first_10_best_artist)

ax.set(xlabel='count of film award', ylabel='artist names')
from matplotlib import pyplot

a4_dims = (20, 12)

fig, ax = pyplot.subplots(figsize=a4_dims)

ax = sns.barplot(x="year_film", y='ceremony',hue ='win',  data=data.head(3000))

from matplotlib import pyplot

a4_dims = (20, 12)

fig, ax = pyplot.subplots(figsize=a4_dims)

ax = sns.barplot(x="year_film", y='ceremony',hue ='win',  data=data.tail(3000))