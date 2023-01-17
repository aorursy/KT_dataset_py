# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())

import numpy as np

from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation

plt.style.use('seaborn-pastel')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
akas = pd.read_csv("../input/imdb-dataset/title.akas.tsv/title.akas.tsv", sep = "\t")

akas.head()
akas.query("title == 'My Hero Academia'")
del akas
ratings = pd.read_csv("../input/imdb-dataset/title.ratings.tsv/title.ratings.tsv", sep = "\t")
episodes = pd.read_csv("../input/imdbdata/data.tsv", sep = "\t")
data_filter = episodes.query("parentTconst == 'tt5626028'")
episodes.head()
ratings.head()
q = """select * from data_filter a

inner join

ratings b

on a.tconst = b.tconst order by seasonNumber, episodeNumber"""

data_filter_1 = pysqldf(q)
data_filter_1 = data_filter_1.drop(69, axis = 0)
data_filter_1['episode_number'] = data_filter_1.reset_index()['index']+1



data_filter_1['episode_number_1'] = 's0'+data_filter_1['seasonNumber']+'e0'+data_filter_1['episodeNumber']
Writer = animation.writers['imagemagick']

writer = Writer(fps=20, metadata=dict(artist='u/bloodlessAcranist'), bitrate=1800)
%matplotlib notebook

fig = plt.figure(figsize=(10,6))

ax = plt.axes(xlim=(0, 70), ylim=(0, 10))

line, = ax.plot([], [], lw=3, color = 'r')



# fig = plt.figure(figsize=(10,6))

# plt.xlim(0,70)

# plt.ylim(np.min(data_filter_2)[0], np.max(data_filter_2)[0])

plt.xlabel('episode', fontsize = 20)

plt.ylabel('Average Ratings', fontsize = 20)

plt.title('My Hero Academia Ratings', fontsize = 20)

# x = data_filter_1.episode_number

# y = data_filter_1.averageRating

# line.set_data(x,y)

def animate(i):

    data = data_filter_1.iloc[:int(i+1),:]

    x = data.episode_number

    y = data.averageRating

    line.set_data(x, y)

    return line,

ani = FuncAnimation(fig, animate, frames=69, repeat = True)

ani.save('MyHeroAcademiaRatings.gif', writer=writer)