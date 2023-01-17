import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import json

import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.animation as animation



import re
#OUTPUT_FILE = "../input/hasthagscambiemos/Hashtags_Cambiemos_28SEP.json"

OUTPUT_FILE = "../input/hashtagsfrente/Hashtags_Frente_28SEP.json"



# Initialize empty list to store tweets

tweets_data = []



# Open connection to file

with open(OUTPUT_FILE, "r") as tweets_file:

    # Read in tweets and store in list

    for line in tweets_file:

        tweet = json.loads(line)

        tweets_data.append(tweet)
df = pd.DataFrame(tweets_data, columns=['user','created_at', 'text'])

df.head()
def populate_tweet_df(tweets):

 

    df['user_id'] = list(map(lambda tweet: tweet['user']['id'], tweets))

 

    df['user_name'] = list(map(lambda tweet: tweet['user']['name'], tweets))

    

    df['location'] = list(map(lambda tweet: tweet['user']['location'], tweets))

 

    df['retweeted_from'] = list(map(lambda tweet: tweet['retweeted_status']['user']['id']

                                  if 'retweeted_status' in tweet.keys() else '', tweets))

 

    df['orignal_text'] = list(map(lambda tweet: tweet['retweeted_status']['text']

                                  if 'retweeted_status' in tweet.keys() else '', tweets))

    

    df['tweet_id'] = list(map(lambda tweet: tweet['retweeted_status']['id']

                                  if 'retweeted_status' in tweet.keys() else '', tweets))

    

    

    return df





df = populate_tweet_df(tweets_data)

df.head()
nodes = df['user_id'].drop_duplicates().dropna()
edges = df[df['retweeted_from']!=''][['user_id', 'retweeted_from']].drop_duplicates()
nodes = pd.merge(nodes, edges.groupby('user_id').count().rename(columns={'retweeted_from': 'out'}), how='left',

            left_on='user_id', right_on='user_id').fillna(0)



nodes = pd.merge(nodes, edges.groupby('retweeted_from').count().rename(columns={'user_id': 'in'}), how='left',

            left_on='user_id', right_on='retweeted_from').fillna(0)



nodes = nodes[nodes['in'] > 0]

nodes = nodes[nodes['out'] > 0]
G = nx.from_pandas_edgelist(edges, 'user_id', 'retweeted_from', create_using=nx.DiGraph())

nx.set_node_attributes(G, pd.Series(nodes['in'].to_list(), index=nodes.user_id).to_dict(), 'in')

nx.set_node_attributes(G, pd.Series(nodes['out'].to_list(), index=nodes.user_id).to_dict(), 'out')
degrees = G.degree()

out_degrees = G.out_degree()

in_degrees = G.in_degree()
with plt.style.context('ggplot'):

    

    plt.loglog(sorted([n[1] for n in list(out_degrees)], reverse=True))

    plt.loglog(sorted([n[1] for n in list(in_degrees)], reverse=True))

    plt.title("Degree rank plot")

    plt.legend(['Out', 'In'])

    plt.ylabel("degree")

    plt.xlabel("rank")
def get_strongly_cc(G, node):

    """ get storngly connected component of node""" 

    for cc in nx.strongly_connected_components(G):

        if node in cc:

            return cc

    else:

        return set()



def get_weakly_cc(G, node):

    """ get weakly connected component of node""" 

    for cc in nx.weakly_connected_components(G):

        if node in cc:

            return cc

    else:

        return set()
SGcc = []

for node in G.nodes():

    strong_component = get_strongly_cc(G, node)  # Weakly connected component of node in G

    if len(strong_component) > len(SGcc):

        SGcc = strong_component
SGcc = G.subgraph(SGcc)
SGcc = SGcc.to_undirected()

SGcc_degree = SGcc.degree()
plt.figure(num=None, figsize=(25, 25), dpi=100, facecolor='w', edgecolor='k')



pos = nx.spring_layout(SGcc)



nx.draw(SGcc, pos, nodelist=dict(SGcc_degree).keys(), node_size=[v * 50 for v in dict(SGcc_degree).values()], 

       width=0.5, alpha=0.5, edge_color='b')



plt.axis('off')

plt.show()
max_degree = sorted(dict(SGcc_degree).values())[-1]

largest_hub = [n for n in dict(SGcc_degree) if SGcc_degree[n]==max_degree]



hub_ego = nx.ego_graph(SGcc, largest_hub[0])



plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')



pos = nx.spring_layout(SGcc)



#nx.draw(hub_ego, pos, node_size=50, width=0.5, alpha=0.5, edge_color='b')

#nx.draw_networkx_nodes(hub_ego, pos, nodelist=largest_hub, node_size=300, node_color='r')



nx.draw(hub_ego, nodelist=hub_ego.nodes(), node_size=[v * 10 for v in dict(hub_ego.degree()).values()], 

       width=0.5, alpha=0.5, edge_color='b')



plt.axis('off')

plt.show()
SGcc.remove_edges_from(SGcc.selfloop_edges())
i=1

while True:

    if len(nx.k_core(SGcc, i)) == 0:

        break

    else:

        print('Core exists for K=%d' % i)

        i += 1
MaxKCore = nx.k_core(SGcc, i-1)

degrees_MaxKCore = MaxKCore.degree()
plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')



pos = nx.spring_layout(MaxKCore)



nx.draw(MaxKCore, pos, nodelist=dict(degrees_MaxKCore).keys(), node_size=[v * 10 for v in dict(degrees_MaxKCore).values()], 

       width=0.5, alpha=0.5, edge_color='b')



plt.axis('off')

plt.show()
def extract(start, tweet):



    words = tweet.split()

    return [word[1:] for word in words if word[0] == start]



def strip_punctuation(s):

    #return s.translate(None, '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

    return s.translate(str.maketrans('','','!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))



def extract_hashtags(tweet):

    # strip the punctuation on the tags you've extracted (directly)

    hashtags = [strip_punctuation(tag) for tag in extract('#', tweet)]

    # hashtags is now a list of hash-tags without any punctuation, but possibly with duplicates



    result = []

    for tag in hashtags:

        if tag not in result:  # check that we haven't seen the tag already (we know it doesn't contain punctuation at this point)

            result.append(tag)

    return result
df['hashtags'] = df['orignal_text'].apply(extract_hashtags)

df2 = df[['orignal_text', 'hashtags']]

df2 = df2[[len(p)>1 for p in df2['hashtags']]]

df2.head()
list_Hashtags = df2['hashtags'].tolist()

                   

H = nx.Graph()



for L in list_Hashtags:

    for i in range(len(L)):

        for j in range(i,len(L)):

            H.add_edge(L[i], L[j])
H = sorted(nx.connected_component_subgraphs(H), key=len, reverse=True)[0]

degrees_h = H.degree()
plt.figure(num=None, figsize=(20, 20), dpi=150, facecolor='w', edgecolor='k')



pos = nx.spring_layout(H)



# nodes

nx.draw_networkx_nodes(H, pos, nodelist=dict(degrees_h).keys(), 

                       node_size=[v * 40 for v in dict(degrees_h).values()], alpha=0.5)



# edges

nx.draw_networkx_edges(H, pos, width=0.3, alpha=0.3, edge_color='b')



# labels

nx.draw_networkx_labels(H, pos, font_size=7, font_family='sans-serif')



plt.axis('off')

plt.show()