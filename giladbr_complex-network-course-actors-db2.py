import pandas as pd

import matplotlib

from itertools import combinations

import networkx as nx # the main libary we will use

from networkx.algorithms import bipartite

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit 

import pandas as pd

import matplotlib

from ast import literal_eval

import collections
movies_df = pd.read_csv('../input/movies_metadata.csv')

credits_df = pd.read_csv('../input/credits.csv')

credits_df.cast = credits_df.cast.apply(literal_eval) #convert str to list of dicts
movies_columns = ['vote_count','vote_average','id','title','original_language','budget','adult','popularity','release_date','revenue',]

movies_df = movies_df.loc[:,movies_columns]

credits_df = credits_df.loc[:,['id','cast']]
movies_df.head()
df = pd.concat([movies_df,credits_df],join='inner',axis=1,keys='id')

df.columns =df.columns.droplevel() #remove multiindex

#keep unadult movies, in english

df = df[(df['adult']=='False') & (df['original_language']=='en')].drop(['adult','original_language'],axis=1)

#convert to numeric

for column in ['popularity','budget','revenue','vote_average','vote_count']:

    df[column]=pd.to_numeric(df[column])

#remove nulls

df=df.dropna()

#keep only year

df['release_date']=df['release_date'].apply(lambda x: int(str(x)[0:4]))

#ranked movies in the 2000's

df = df[(df['vote_average']>0) & (df.release_date>2010) &(df.budget>0) ]

#remove duplicate id column

df = df.loc[:,~df.columns.duplicated()]

#keep only actor list

df.cast = df.cast.apply(lambda x: [d['name'] for d in x])
df.shape
#Graph preproccesing



df['cast_coup']=df['cast'].apply(lambda L: [comb for comb in combinations(L, 2)])

df['id_tup']=df['id'].apply(lambda x: (str(x),))



def generate_touples(tup,row):

    return tup+row['id_tup']



df['cast_coup_step2']=df.apply(lambda row: [generate_touples(tup,row) for tup in row['cast_coup']],axis=1)

final_tup_list = df['cast_coup_step2'].apply(pd.Series).stack().reset_index(True)[0].tolist()

df_players = pd.DataFrame(final_tup_list, columns=['Player1', 'Player2', 'id'])

df_players.head()
df_columns = ['vote_count','vote_average','id','title','budget','popularity','release_date','revenue',]

df_final = df_players.merge(df.loc[:,df_columns], on='id')

df_final.sample(5,random_state=100)

df_final.to_csv('df_final.csv',index=False)
G = nx.from_pandas_edgelist(df_final,'Player1','Player2',

                            edge_attr = ['vote_count','title','budget','popularity','release_date','revenue','vote_average'], 

                            create_using=nx.Graph())

print (nx.info(G))
nx.write_gexf(G, "G.gexf")
def Bacon_number(G,Player1,Player2):

        length =nx.shortest_path_length(G,Player1,Player2)

        path = nx.shortest_path(G,Player1,Player2) # get the length of shortest path between 2 nodes (d)

        attr_year = nx.get_edge_attributes(G,'release_date')

        attr_name = nx.get_edge_attributes(G,'title')

        print('%s Bacon number to actor %s is %d' %(Player1,Player2,length))

        for edge in range(length):

            try:

                print('At the year of %s, %s and %s Played in a movie called %s' %(str(attr_year[(path[edge],path[edge+1])]),path[edge],path[edge+1], attr_name[(path[edge],path[edge+1])]))

            except:

                print('At the year of %s, %s and %s Played in a movie called %s' %(str(attr_year[(path[edge+1],path[edge])]),path[edge],path[edge+1], attr_name[(path[edge+1],path[edge])]))

            #attr_year[(path[edge],path[edge+1])],attr_name[(path[edge],path[edge+1])]



# Bacon_number(G,'Brad Pitt', 'Gal Gadot')
#P(k) degree_distribution

def degree_distribution(G):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence

    # print "Degree sequence", degree_sequence

    degreeCount = collections.Counter(degree_sequence)

    deg, cnt = zip(*degreeCount.items())

    return np.sum(np.array(deg)*np.array(cnt))/sum(cnt)

# i_avg = degree_distribution(G)
# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence

# # print "Degree sequence", degree_sequence

# degreeCount = collections.Counter(degree_sequence)

# deg, cnt = zip(*degreeCount.items())



# fig, ax = plt.subplots()

# plt.bar(deg, cnt, width=0.80, color='b')



# plt.title("Degree Histogram")

# plt.ylabel("Count")

# plt.xlabel("Degree")

# ax.set_xticks([d + 0.4 for d in deg])

# ax.set_xticklabels(deg)



# # draw graph in inset

# plt.axes([0.4, 0.4, 0.5, 0.5])

# Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]

# pos = nx.spring_layout(G)

# plt.axis('off')

# nx.draw_networkx_nodes(G, pos, node_size=20)

# nx.draw_networkx_edges(G, pos, alpha=0.4)



# plt.show()
# print('The average path length <i> is %0.2f' % (nx.average_shortest_path_length(G)))
# #average path length

# def i_avg(G):

#     a=list()

#     for C in nx.connected_component_subgraphs(G):

#         a.append(nx.average_shortest_path_length(C))

#     return avg(a)



# print('The average path length <i> is %0.2f' % (i_avg(G)))
# import holoviews as hv

# hv.extension('bokeh')

# hv.Graph.from_networkx(G, nx.layout.spring_layout).options(

# edge_color_index='vote_average',width=750, height=500,inspection_policy='edges',directed=True,arrowhead_length=0.01)

# Measure: N(t), L(t) [t- time if you have a time dependent system);  

 

#  <l> average path length; 

#  C (clustering coefficient), Crand, C(k);

# Degree correlation(Knn vs k), Hierarchy (C(k) vs k)

# Visualization/communities; 

# P(w) if you have a weighted network; 

# Spreading effect (if appropriate).

# print(nx.is_connected(G))

# nx.k_nearest_neighbors(G)
