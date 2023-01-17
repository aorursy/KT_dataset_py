import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

from itertools import chain
netflix = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv")

netflix.head()

movies = netflix.loc[netflix["type"]=="Movie"]

shows = netflix.loc[netflix["type"]=="TV Show"]

movies.info()

shows.info()
n=10

import matplotlib.pyplot as plt 

top_directors = movies.director.str.split(',', expand = True).stack().value_counts()[:n]

y_pos = np.arange(len(top_directors.index))

plt.bar(top_directors.index, top_directors.array, color='green')

plt.xticks(y_pos, top_directors.index,rotation = "vertical")

plt.yticks(np.arange(19)[::2])

plt.title("Top 10 Movie Directors")

plt.ylabel("Number of Titles")

plt.show()
top_directors =shows.director.str.split(',', expand = True).stack().value_counts()[:n]

plt.bar(top_directors.index, top_directors.array, color='green')

plt.xticks(y_pos, top_directors.index,rotation = "vertical")

plt.yticks(np.arange(5)[::2])

plt.title("Top 10 TV Show Directors")

plt.ylabel("Number of Titles")

plt.show()
top_actors = movies.cast.str.split(',', expand = True).stack().value_counts()[:n]

plt.bar(top_actors.index, top_actors.array, color='green')

plt.xticks(y_pos, top_actors.index,rotation = "vertical")

plt.yticks(np.arange(28)[::2])

plt.title("Top 10 Movies actors")

plt.ylabel("Number of Titles")

plt.show()
top_actors = shows.cast.str.split(',', expand = True).stack().value_counts()[:n]

plt.bar(top_actors.index, top_actors.array, color='green')

plt.xticks(y_pos, top_actors.index,rotation = "vertical")

plt.yticks(np.arange(18)[::2])

plt.title("Top 10 TV actors")

plt.ylabel("Number of Titles")

plt.show()
#generate the list of actors

american_movies = netflix[(netflix["country"].str.contains("United States"))& (netflix["type"]=="Movie")]

split_cast = american_movies.cast.str.split(',', expand = True)

actors = []

for i in range(len(split_cast.columns)):

    actors.extend(split_cast[i].unique())



actors1 = list(set(actors)) #remove duplicates

actors1.pop(0); #remove none

actors1.remove(None) #remove None

actors =[]

for i in actors1: #remove spaces from beginning and end of name

    j = i.strip()

    actors.append(j)
#make a list of actors that have been in a movie with another actor

connections = []

movies_dict ={}  #to store what movies they were in will be added as attributes to a network

for i in range(len(actors)):

    con = []

    appears = np.where(american_movies['cast'].str.contains(actors[i])==1) #find where they appear

    reindex_movies = american_movies['cast'].reset_index(drop=True) #reindex the movies to match appears

    reindex_titles = american_movies['title'].reset_index(drop = True) #reindex movie titles

    movies_dict[actors[i]] = reindex_titles[appears[0]]

    for ii in range(len(appears[0])): #make a list of connections for each person

        list_acts = reindex_movies[appears[0][ii]]

        con.extend(list_acts.split(","))

    cons = []

    for ii in con: #remove spaces on names

        j = ii.strip()

        cons.append(j)

    cons = list(set(cons)) #remove duplicates

    cons.remove(actors[i]) #remove the queried actor from the list

    connections.append(cons)
#make a web with the actors that have appears in other movies

import networkx as nx

from operator import itemgetter

import community



G = nx.Graph()

G.add_nodes_from(actors) # Add nodes to the Graph                             

edges = []

for i in range(len(actors)):

    for ii in range(len(connections[i])):

        edges.append((actors[i],connections[i][ii]))

G.add_edges_from(edges) #add edges to the graph

nx.set_node_attributes(G, movies_dict, 'Movies')
#see if two actors are connected

j = 1500

k = 840

if nx.has_path(G, actors[j],actors[k])==True:

    print(actors[j] + " and " + actors[k] + " are connected. It takes " +

          str(nx.shortest_path_length(G,source=actors[j],target=actors[k])) + " movies to move between them.")

    path = nx.shortest_path(G, source= actors[j], target = actors[k])

    for i in range(len(path)-1):

        movies1 = G.nodes[path[i]]["Movies"]

        movies2 = G.nodes[path[i+1]]["Movies"]

        if i < len(path)-2:

            print(path[i] + ' was in ' + str(set(movies1).intersection(movies2)) + ' with '+ path[i+1])

        else:

            print(path[i] + ' was in ' + str(set(movies1).intersection(movies2)) + ' with '+ path[i+1] + '.')

            

else:

    print(actors[j] + " and " + actors[k] + " are not connected.")
#if you want to specify the actors 

j = "Daniel Craig"

k = "Brent Werzner"

if nx.has_path(G, j , k)==True:

    print(j + " and " + k + " are connected. It takes " +

          str(nx.shortest_path_length(G,source=j,target=k)) + " movies to move between them.")

    path = nx.shortest_path(G, source= j, target = k)

    for i in range(len(path)-1):

        movies1 = G.nodes[path[i]]["Movies"]

        movies2 = G.nodes[path[i+1]]["Movies"]

        if i < len(path)-2:

            print(path[i] + ' was in ' + str(set(movies1).intersection(movies2)) + ' with '+ path[i+1])

        else:

            print(path[i] + ' was in ' + str(set(movies1).intersection(movies2)) + ' with '+ path[i+1] + '.')

            

else:

    print(j + " and " + k + " are not connected.")
print("The following actors " + str([n for n in nx.isolates(G)]) + " have not been in a movie with anyone else.")