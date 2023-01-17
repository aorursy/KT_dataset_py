# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv("../input/movie_metadata.csv")

movies.head()
movies['revenue']=movies['gross']-movies['budget']

movies.head()
corrcol = ['director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','cast_total_facebook_likes','movie_facebook_likes','num_voted_users','num_user_for_reviews','imdb_score','revenue']

mvcorr = movies[corrcol]

mvcorr = mvcorr[mvcorr.director_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.actor_1_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.actor_2_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.cast_total_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.movie_facebook_likes != 0]

correlation = mvcorr[corrcol].corr(method='pearson')

#correlation.to_csv('csv/correlation.csv',encoding='utf-8')

fig, axes = plt.subplots()

sns.heatmap(correlation, annot=True)

plt.show()

plt.close()
topdir = movies.sort_values(by='director_facebook_likes', ascending=0)

topdir = topdir[['director_name','director_facebook_likes']]

topdir = topdir.drop_duplicates()[:50]

topdir = topdir.set_index('director_name')

topdir.index
topdir2 = movies[['director_name','imdb_score']]

topdir2 = topdir2.groupby(['director_name']).mean()

topdir2 = topdir2.dropna()

topdir2 = topdir2.sort_values(by='imdb_score', ascending=0)[:50]

topdir2.index
topdir3 = movies[['director_name','revenue']]

topdir3 = topdir3.groupby(['director_name']).mean()

topdir3 = topdir3.dropna()

topdir3 = topdir3.sort_values(by='revenue', ascending=0)[:50]

topdir3.index
bestdir = topdir.index.intersection(topdir2.index)

bestdir = bestdir.intersection(topdir3.index)

bestdir
from matplotlib_venn import venn3, venn3_circles

set1 = set(topdir.index.values)

set2 = set(topdir2.index.values)

set3 = set(topdir3.index.values)

v = venn3([set1, set2, set3], ('Top 50 FB Likes', 'Top 50 Avg. IMDB Score', 'Top 50 Avg. Revenue'))

plt.title("Who is the best director ?", fontsize=16,fontweight='bold',family='serif')

bestdirstr = ''.join(bestdir)

for text in v.set_labels:

    text.set_fontsize(12)

    text.set_family('serif')

plt.annotate(bestdirstr,fontsize=12,family='serif',xy=v.get_label_by_id('111').get_position()- np.array([0, 0.05]), xytext=(-50,-50),

             ha='center',textcoords='offset points',bbox=dict(boxstyle='round,pad=0.5',fc='white'),

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='black'))

plt.show()