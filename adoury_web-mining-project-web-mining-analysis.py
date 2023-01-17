import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import networkx as nx



import random

random.seed(1234)



import community

import itertools

from collections import defaultdict



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



data_dir = '../input/data_competition/data_competition' 



relation_users = pd.read_csv(data_dir + "/UserUser.txt", sep="\t", header=None)

relation_users.columns = ["follower", "followed"]



labels_training = pd.read_csv(data_dir + "/labels_training.txt", sep=",")

labels_training.columns = ["news", "label"]



news_users = pd.read_csv(data_dir + "/newsUser.txt", sep="\t", header=None)

news_users.columns = ["news", "user", "times"]

G = nx.DiGraph()



edges = [tuple(x) for x in relation_users.values]

G.add_edges_from(edges)



nx.info(G)
n_edges = G.number_of_edges()

n_nodes = G.number_of_nodes()

news_user_lab = pd.merge(news_users, labels_training, on='news')



user_sum_lab = news_user_lab[['user', 'label','times']].groupby(['user', 'label'])['times'].sum().reset_index()

user_sum_lab.sort_values(['times'], ascending=[False])



user_tot = news_users[['user', 'times']].groupby(['user'])['times'].sum().reset_index()

user_tot.set_index('user', inplace = True)



user_perc_lab = user_sum_lab[['user', 'times']].groupby(['user'])['times'].sum().reset_index()



user_perc_fake = pd.merge(user_perc_lab, user_sum_lab[user_sum_lab["label"] == 1], on='user')

user_perc_fake.columns = ["user", "total_nb", "label", "fake_nb"]

user_perc_fake = user_perc_fake[["user", "total_nb", "fake_nb"]]

user_perc_fake['perc_fake'] = user_perc_fake["fake_nb"]/user_perc_fake["total_nb"]

user_perc_fake = user_perc_fake.sort_values(['total_nb', 'perc_fake'], ascending=[False, True])

user_perc_fake.set_index('user', inplace=True)

print(user_perc_fake.head())

print(user_perc_fake.tail())
print(len(user_sum_lab))

print(len(user_perc_fake))
for node in G.nodes():

    if user_tot.times[node] < 5 :

        G.nodes[node]['main_user'] = 0

    else :

        G.nodes[node]['main_user'] = 1

    

for node in G.nodes():

    if node in user_perc_fake.index:

        G.nodes[node]['fake_source_level'] = (user_perc_fake.perc_fake[node]>0.5)*1 

    elif node in user_sum_lab['user'] :

        G.nodes[node]['fake_source_level'] = 0

    else:

        G.nodes[node]['fake_source_level'] = -1

fake_source = []

real_source = []

main_fake = []

main_real = []



for node in G.nodes():

    if G.nodes[node]['fake_source_level'] == 1:

        fake_source.append(node)

        if G.nodes[node]['main_user'] == 1:

            main_fake.append(node)

    elif G.nodes[node]['fake_source_level'] == 0:

        real_source.append(node)

        if G.nodes[node]['main_user'] == 1:

            main_real.append(node)

        

print(len(real_source))

print(len(fake_source))

print(len(main_fake))

print(len(main_real))

G_bigU = nx.DiGraph(G)

remove = [node for node in G_bigU.nodes() if G_bigU.nodes[node]['main_user'] == 0 ]

G_bigU.remove_nodes_from(remove)



print(nx.info(G_bigU))

print(G_bigU.number_of_nodes())

color_node = list()



for i in G_bigU.nodes():

    color_node.append(G_bigU.nodes[i]['fake_source_level']+1)





options = {

    'node_color' : color_node, # a list that contains the community id for the nodes we want to plot

    'node_size' : 30 , 

    'cmap' : plt.get_cmap("jet"),

    'node_shape' : 'o', 

    "width" : 0.1, 

    "font_size" : 15,

    "alpha" : 0.8   

}

nx.draw(G_bigU,  **options)

plt.figure(figsize=(40,40))

plt.show()
### We set G_u and G_main_user_und as the undirected versions of G and G_bigU



G_u = G.to_undirected()

G_main_user_und = G_bigU.to_undirected()
connected_comp =  list(nx.connected_component_subgraphs(G_u))

l_con = []

for i in connected_comp:

    l_con.append(len(i))

print(l_con)
G_connected = connected_comp[0]

nx.info(G_connected)
centrality =  nx.degree_centrality(G)
real_cd = []

fake_cd = []

main_fake_cd =[]

main_real_cd =[]

for x in real_source:

    real_cd.append(centrality[x])

for x in fake_source:

    fake_cd.append(centrality[x])

for x in main_fake:

    main_fake_cd.append(centrality[x])

for x in main_real:

    main_real_cd.append(centrality[x])

    

sns.distplot(fake_cd)

plt.title('Degree Centrality for "Fake" Sources')

plt.show()





sns.distplot(real_cd)

plt.title('Degree Centrality for "Real" Sources')

plt.show()





sns.distplot(main_fake_cd)

plt.title('Degree Centrality for main fake sources')

plt.show()



sns.distplot(main_real_cd)

plt.title('Degree Centrality for main real sources')

plt.show()
in_centrality = nx.in_degree_centrality(G)

real_in_cd = []

fake_in_cd = []

main_fake_incd = []

main_real_incd = []



for x in real_source:

    real_in_cd.append(in_centrality[x])

for x in fake_source:

    fake_in_cd.append(in_centrality[x])

for x in main_fake:

    main_fake_incd.append(in_centrality[x])

for x in main_real:

    main_real_incd.append(in_centrality[x])

    

    

sns.distplot(fake_in_cd)

plt.title('In-degree Centrality for "Fake" Sources')

plt.show()



sns.distplot(real_in_cd)

plt.title('In-degree Centrality for "Real" Sources')

plt.show()



sns.distplot(main_fake_incd)

plt.title('In-degree Centrality for main fake sources')

plt.show()



sns.distplot(main_real_incd)

plt.title('In-degree Centrality for main real sources')

plt.show()
out_centrality = nx.out_degree_centrality(G)



real_out_cd = []

fake_out_cd = []

main_fake_outcd = []

main_real_outcd = []



for x in real_source:

    real_out_cd.append(out_centrality[x])

for x in fake_source:

    fake_out_cd.append(out_centrality[x])

for x in main_fake:

    main_fake_outcd.append(out_centrality[x])

for x in main_real:

    main_real_outcd.append(out_centrality[x])

    

sns.distplot(fake_out_cd)

plt.title('Out-degree Centrality for "Fake" Sources')

plt.show()





sns.distplot(real_out_cd)

plt.title('Out-degree Centrality for "Real" Sources')

plt.show()





sns.distplot(main_fake_outcd)

plt.title('Out-degree Centrality for main fake sources')

plt.show()



sns.distplot(main_real_outcd)

plt.title('Out-degree Centrality for main real sources')

plt.show()
eig_centrality = nx.eigenvector_centrality(G)



real_eig_cd = []

fake_eig_cd = []

main_fake_eig = []

main_real_eig = []



for x in real_source:

    real_out_cd.append(eig_centrality[x])

for x in fake_source:

    fake_out_cd.append(eig_centrality[x])

for x in main_fake:

    main_fake_outcd.append(eig_centrality[x])

for x in main_real:

    main_real_outcd.append(eig_centrality[x])

    

sns.distplot(fake_out_cd)

plt.title('Eigenvector Centrality for "Fake" Sources')

plt.show()





sns.distplot(real_out_cd)

plt.title('Eigenvector Centrality for "Real" Sources')

plt.show()





sns.distplot(main_fake_outcd)

plt.title('Eigenvector Centrality for main fake sources')

plt.show()



sns.distplot(main_real_outcd)

plt.title('Eigenvector Centrality for main real sources')

plt.show()
page_rank = nx.pagerank(G)
real_pgr = []

fake_pgr= []

main_fake_pgr = []

main_real_pgr = []



for x in real_source:

    real_pgr.append(page_rank[x])

for x in fake_source:

    fake_pgr.append(page_rank[x])

for x in main_fake:

    main_fake_pgr.append(page_rank[x])

for x in main_real:

    main_real_pgr.append(page_rank[x])

    

sns.distplot(fake_pgr)

plt.title('PageRank density for "Fake" Sources')

plt.show()





sns.distplot(real_pgr)

plt.title('PageRank density for "Real" Sources')

plt.show()



sns.distplot(main_fake_pgr)

plt.title('PageRank density for main fake sources')

plt.show()



sns.distplot(main_real_pgr)

plt.title('PageRank density for main real sources')

plt.show()



metrics_user =[centrality, in_centrality, out_centrality, page_rank]

d={}

for k in centrality.keys():

    d[k] = tuple(d[k] for d in metrics_user)

metrics_user=pd.DataFrame.from_dict(d,orient='index',columns=['degree_centrality','in_degree_centrality',

                                                       'out_degree_centrality', 'page_rank'])

metrics_user.index.rename('user')
metrics_user_news = pd.merge(metrics_user,news_users.set_index('user'), left_index=True, right_index=True)

metrics_user_news.index.name= 'user'

metrics_news = metrics_user_news.groupby('news').agg(['sum','mean','max','min'])

metrics_news = metrics_news.drop(columns=[('times', 'mean'),('times', 'max'),('times', 'min')])

#metrics_news.to_csv('metrics_news.csv',index=False)
metrics_news.head(10)
partition = community.best_partition(G_u, random_state=1234)



v = defaultdict(list)



for key, value in sorted(partition.items()):

    v[value].append(key)

nb_fake = []

nb_real = []

nb_fake_main = []

nb_real_main = []

nb_small = []



nb_test = []



l = []

N=0

for i in v.keys():

    l.append(len(v[i]))

    nb_fake.append(0)

    nb_real.append(0)

    nb_fake_main.append(0)

    nb_real_main.append(0)

    nb_test.append(0)

    nb_small.append(0)

    for j in v[i]:

        if G.nodes[j]['fake_source_level']==0 :

            nb_real[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_real_main[N] +=1

        elif G.nodes[j]['fake_source_level']==1 :

            nb_fake[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_fake_main[N] +=1

        elif  G.nodes[j]['fake_source_level']==-1:

            nb_test[N] +=1

        if  G.nodes[j]['main_user']==0:

            nb_small[N] +=1

    N += 1



df_louvain = pd.DataFrame({'length': l, 

                  'nb_fake' : nb_fake,

                  'nb_real' : nb_real,

                  'nb_main_fake': nb_fake_main,

                  'nb_main_real': nb_real_main,

                  'nb_small_users': nb_small,

                  'unknown credibility' : nb_test},

                          columns =['length','nb_fake','nb_real','nb_main_fake','nb_main_real','nb_small_users','unknown credibility'])

df_louvain
print('There are %1.f communities with more than 10 users ' %df_louvain.length[df_louvain["length"]>=10].count())

print('There are %1.f communities with less than 5 users ' %df_louvain.length[df_louvain["length"]<5].count())
### Communities with more than 10 nodes



df_louvain[df_louvain["length"]>=10]
#Proportion of users in each category for the communities with more than 10 nodes



df_louvain[['length','nb_fake','nb_real','nb_main_fake','nb_main_real']][df_louvain["length"]>=10].div(df_louvain.length[df_louvain["length"]>=10], axis=0)
partition2 = community.best_partition(G_connected)

v = defaultdict(list)



for key, value in sorted(partition2.items()):

    v[value].append(key)

    

nb_fake = []

nb_real = []

nb_fake_main = []

nb_real_main = []

nb_small = []



nb_test = []



l = []

N=0

for i in v.keys():

    l.append(len(v[i]))

    nb_fake.append(0)

    nb_real.append(0)

    nb_fake_main.append(0)

    nb_real_main.append(0)

    nb_test.append(0)

    nb_small.append(0)

    for j in v[i]:

        if G.nodes[j]['fake_source_level']==0 :

            nb_real[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_real_main[N] +=1

        elif G.nodes[j]['fake_source_level']==1 :

            nb_fake[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_fake_main[N] +=1

        elif  G.nodes[j]['fake_source_level']==-1:

            nb_test[N] +=1

        if  G.nodes[j]['main_user']==0:

            nb_small[N] +=1

    N += 1



df_louvain2 = pd.DataFrame({'length ': l, 

                  'nb_fake' : nb_fake,

                  'nb_real' : nb_real,

                  'nb_main_fake': nb_fake_main,

                  'nb_main_real': nb_real_main,

                  'nb_small_users': nb_small,

                  'unknown credibility' : nb_test} )

df_louvain2
from networkx.algorithms import community



b=community.asyn_fluidc(G_connected, 2)

a=community.coverage(G_connected,list(b))

best = 2

for k in [4,6,8,10]:

    fluid=community.asyn_fluidc(G_connected, k)

    c = community.coverage(G_connected, list(fluid))

    print(k,c)

    if c>a:

        a=community.coverage(G_connected,list(fluid))

        best=k

        

print(best)



Fcommunities = list(community.asyn_fluidc(G_connected, 2))

for i in Fcommunities:

    print(len(i))
nb_fake = []

nb_real = []

nb_fake_main = []

nb_real_main = []

nb_small = []



nb_test = []



l = []

N=0

for i in Fcommunities:

    l.append(len(i))

    nb_fake.append(0)

    nb_real.append(0)

    nb_fake_main.append(0)

    nb_real_main.append(0)

    nb_test.append(0)

    nb_small.append(0)

    for j in i:

        if G.nodes[j]['fake_source_level']==0 :

            nb_real[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_real_main[N] +=1

        elif G.nodes[j]['fake_source_level']==1 :

            nb_fake[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_fake_main[N] +=1

        elif  G.nodes[j]['fake_source_level']==-1:

            nb_test[N] +=1

        if  G.nodes[j]['main_user']==0:

            nb_small[N] +=1

    N += 1







df = pd.DataFrame({'length ': l, 

                  'nb_fake' : nb_fake,

                  'nb_real' : nb_real,

                  'nb_main_fake': nb_fake_main,

                  'nb_main_real': nb_real_main,

                  'nb_small_users': nb_small,

                  'unknown credibility' : nb_test} )

df
Fcommunities = list(community.asyn_fluidc(G_connected, 6))

for i in Fcommunities:

    print(len(i))
nb_fake = []

nb_real = []

nb_fake_main = []

nb_real_main = []

nb_small = []



nb_test = []



l = []

N=0

for i in Fcommunities:

    l.append(len(i))

    nb_fake.append(0)

    nb_real.append(0)

    nb_fake_main.append(0)

    nb_real_main.append(0)

    nb_test.append(0)

    nb_small.append(0)

    for j in i:

        if G.nodes[j]['fake_source_level']==0 :

            nb_real[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_real_main[N] +=1

        elif G.nodes[j]['fake_source_level']==1 :

            nb_fake[N] += 1

            if G.nodes[j]['main_user']==1:

                nb_fake_main[N] +=1

        elif  G.nodes[j]['fake_source_level']==-1:

            nb_test[N] +=1

        if  G.nodes[j]['main_user']==0:

            nb_small[N] +=1

    N += 1







df_fcommu = pd.DataFrame({'length ': l, 

                  'nb_fake' : nb_fake,

                  'nb_real' : nb_real,

                  'nb_main_fake': nb_fake_main,

                  'nb_main_real': nb_real_main,

                  'nb_small_users': nb_small,

                  'unknown credibility' : nb_test} )

df_fcommu
Pcommunities = list(community.k_clique_communities(G_u, 5, cliques=None))
print(len(Pcommunities))

l=[]

for i in Pcommunities:

    l.append(len(i))