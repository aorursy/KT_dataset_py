# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#import dataset

dataset=pd.read_csv('/kaggle/input/results.csv')

dataset.head(5)
countries=[]



for c1, c2 in zip(dataset['home_team'],dataset['away_team']):

    if c1 not in countries :

        countries.append(c1)

    if c2 not in countries :

        countries.append(c2)





team_vict=[]

for c1,(j,c2) in zip(dataset['home_team'],enumerate(dataset['away_team'])):

    if dataset['home_score'][j]>dataset['away_score'][j]:

        team_vict.append(c1)

    elif (dataset['home_score'][j]<dataset['away_score'][j]):

        team_vict.append(c2)

    else:

        team_vict.append('draw')

team_vict=[i for i in team_vict if i!='draw']
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=0 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(team_vict)

matches=[]

for c in countries:

    nb=0

    for c1, c2 in zip(dataset['home_team'],dataset['away_team']):

            if c1==c or c2==c:

                nb=nb+1

    matches.append(nb)

        

victoires=[]

defaites=[]

nuls=[]

pourcent_vic=[]

for c in countries:

    nb_v=nb_d=nb_n=0

    for c1 ,(j ,c2) in zip(dataset['home_team'],enumerate(dataset['away_team'])):

        if c1 == c:

            if dataset['home_score'][j]>dataset['away_score'][j]:

                nb_v=nb_v+1

            elif dataset['home_score'][j]<dataset['away_score'][j]:

                nb_d=nb_d+1

            else:

                nb_n=nb_n+1

        elif c2 == c:

            if dataset['away_score'][j]>dataset['home_score'][j]:

                nb_v=nb_v+1

            elif dataset['away_score'][j]<dataset['home_score'][j]:

                nb_d=nb_d+1

            else:

                nb_n=nb_n+1

    victoires.append(nb_v)

    defaites.append(nb_d)

    nuls.append(nb_n)

statistics=pd.DataFrame(list(zip(countries,matches,victoires,defaites,nuls)),columns=['country','nb_match','nb_vict',

                                                                                     'nb_defts','nb_draws'])

statistics['%vic']=round((statistics['nb_vict']/statistics['nb_match'])*100)

statistics.head()
X=statistics.iloc[:,2:-1].values
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 16):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 16), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
my_cluster=KMeans(n_clusters = 4 ,init = 'k-means++')

y_kmeans=my_cluster.fit_predict(X)
def worcloud_clusters(corpus,clusters,n_clusters):

    for i in range(n_clusters):

        corps=[]

        print('cluster numéro',i+1)

        for j in range (len(clusters)):

            if clusters[j]==i:

                corps.append(corpus[j])

        if len(corps)!=0:

            show_wordcloud(corps)
worcloud_clusters(statistics.country,y_kmeans,4)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],X[y_kmeans == 0, 2], s = 100, c = 'red', label ='weak teams')

ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],X[y_kmeans == 1, 2], s = 100, c = 'blue', label = 'standars teams 1')

ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 1],s = 100, c = 'green', label = 'standars teams 2')

ax.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], X[y_kmeans == 3, 1],s = 100, c = 'cyan', label = 'top teams')

ax.set_title('Clusters of teams')

ax.set_xlabel('victs')

ax.set_xlabel('defts')

ax.set_xlabel('draws')

plt.legend()

plt.show()
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label ='weak teams')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'standars teams 1')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'standars teams 2')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'top teams')

plt.title('Clusters of teams')

plt.xlabel('nb of vict')

plt.ylabel('nb of defts')

plt.legend()

plt.show()