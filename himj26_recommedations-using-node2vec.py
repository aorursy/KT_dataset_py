!pip install node2vec


import numpy as np 

import pandas as pd

import networkx as nx #create and store graph

from node2vec import Node2Vec #To run node2vec algorithm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('../input/netflix-shows/netflix_titles.csv')

df=df.dropna()

df=df.drop(['description'],axis=1)

df.head()
df['title'].is_unique
df2=df.groupby(['title']).count()

print(df2[df2['show_id']>1][0:2]) 

#Print the lists of titles appearing more than one time
df[df['title']=='Benji']

# Since there are same titles appaearing in different years we will combine year and title
df['title']=df['title']+', '+df['date_added']

df['title'].is_unique
df[df['title']=='The Silence, March 1, 2018']

#Now they are simply repeated with exactly we will drop them
df.drop_duplicates(subset=['title'],keep = False, inplace = True) 

df['title'].is_unique
df.head()
# function that will create edges for given movie title and its genres

def addToGraph(movie_name,graph):

    genres=df[df['title']==movie_name]['listed_in'].values[0].rstrip().lower().split(', ')

    for genre in genres:

        graph.add_edge(movie_name.strip(),genre)

    return graph



#function that will create graph for all the movies name

def createGraph():

    graph = nx.Graph()

    for movie_name in df['title']:

        graph=addToGraph(movie_name,graph)

    return graph
graph=createGraph()
print(graph.degree()['Norm of the North: King Sized Adventure, September 9, 2019']) #should be 2 since two genres are assoicated with it

print(graph.degree()['#realityhigh, September 8, 2017']) #shoukd be 1 since 1 genres are assoicated with it
node2vec = Node2Vec(graph, dimensions=20, walk_length=16, num_walks=10)
model = node2vec.fit(window=5, min_count=1)
model.wv.get_vector('Ralph Breaks the Internet: Wreck-It Ralph 2, June 11, 2019')
model.wv.get_vector('Transformer, February 20, 2019')
#generate similiar movies to given genre or title

def print_similiar(name):

    for node, _ in model.wv.most_similar(name):

        print(node)
print_similiar('children & family movies')

#As we can see most movies like "Barbie", "sKrish Trish and Baltiboy" are indeed children movies and shows
print_similiar('Naruto Shippuden : Blood Prison, September 1, 2017')

# We get results like another naruto series, seven deadly sins which are really good recommendation.