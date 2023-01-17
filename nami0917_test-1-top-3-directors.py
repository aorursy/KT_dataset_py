# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from matplotlib_venn import venn3, venn3_circles



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.read_csv("../input/movie_metadata.csv")



movies = pd.read_csv("../input/movie_metadata.csv")

movies.shape
movies = pd.read_csv("../input/movie_metadata.csv")

movies.head()
movies.describe()
movies.corr() # corr of all data 
df = pd.DataFrame(movies)

movies2 =df.dropna() # new data without NA

movies2.head(10)

movies2.describe() 
movies2.corr() # corr of new data
correlation = movies2.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different variables')

movies2['revenue']=movies2['gross']-movies2['budget']  

movies2.head(10)
corrcol = ['num_critic_for_reviews','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','cast_total_facebook_likes','movie_facebook_likes','num_voted_users','num_user_for_reviews','imdb_score','revenue']

mvcorr = movies2[corrcol]

mvcorr = mvcorr[mvcorr.num_critic_for_reviews != 0]

mvcorr = mvcorr[mvcorr.director_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.actor_1_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.actor_2_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.cast_total_facebook_likes != 0]

mvcorr = mvcorr[mvcorr.movie_facebook_likes != 0]

correlation2 = mvcorr[corrcol].corr(method='pearson')



plt.figure(figsize=(10,10))

sns.heatmap(correlation2, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different variables 2')
topdirector1 = movies2[['director_name','revenue']]

topdirector1 = topdirector1.groupby(['director_name']).mean()

topdirector1 = topdirector1.sort_values(by='revenue', ascending=0)[:100]

topdirector1.index
topdirector2 = movies2[['director_name','imdb_score']]

topdirector2 = topdirector2.groupby(['director_name']).mean()

topdirector2 = topdirector2.sort_values(by='imdb_score', ascending=0)[:100]

topdirector2.index
topdirector3 = movies2[['director_name','movie_facebook_likes']]

topdirector3 = topdirector3.groupby(['director_name']).mean()

topdirector3 = topdirector3.sort_values(by='movie_facebook_likes', ascending=0)[:100]

topdirector3.index
bestdirector = topdirector1.index.intersection(topdirector2.index)

bestdirector = bestdirector.intersection(topdirector3.index)

bestdirector = bestdirector.sort_values(ascending=0)[:3]

bestdirector
set1 = set(topdirector1.index.values)

set2 = set(topdirector2.index.values)

set3 = set(topdirector3.index.values)

v = venn3([set1, set2, set3], ('Top 100 Average Revenue','Top 100 Average IMDB Score','Top 100 Facebook Likes'))

bestdirectorstr = ''+", ".join(bestdirector)



plt.annotate(bestdirectorstr,fontsize=15,family='serif',xy=v.get_label_by_id('111').get_position()- np.array([0, 0.15]), xytext=(-50,-50),

             ha='center',textcoords='offset points',bbox=dict(boxstyle='round,pad=0.5',fc='white'),

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='black'))

plt.title("Top 3 Directors", fontsize=16,fontweight='bold',family='serif')



plt.show()