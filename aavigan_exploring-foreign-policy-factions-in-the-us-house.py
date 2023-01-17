from ast import literal_eval

import pandas as pd

import networkx as nx

from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt

import community



from collections import defaultdict

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def plot_wordcloud(data, index, community):

    

    #text includes all the titles of bills sponsored by the community of interest

    text = ' '.join([str(row.title)  for i, row in data.loc[index].iterrows()])



    #add relevant words to ignore from wordcloud plot

    stopwords = set(STOPWORDS)

    stopwords.update(['international', 'foreign','government', 'national', 'congressional', 'americans', 'state', 'department', 'congress','united', 'states','condemning', 'including', 'related',

                     'various', 'bill', 'direct', 'countries', 'certain', 'report','president', 'efforts', 'provide', 'directs', 'nan', 'group', 'person','act','individual',

                     'whether', 'entitites', 'groups', 'government', 'include', 'annually', 'issue', 'citizen', 'country', 'assessment',

                     'activities', 'issues', 'service', 'used', 'purposes','security', 'expressing', 'sense', 'people', 'support',

                     'require', 'reaffirming', 'authorize', 'affirming', 'secretary', 'recognizing', 'republic', 'organization', 'organizations',

                     'designation', 'People', 'authorizes', 'establish', 'resolution', 'House', 'Representatives', 'promote', 'enhance','respect', 'prohibit', 'day', 'amend', 'calling', 'impose', 'prevent'])

    # Create and generate a word cloud image:

    wordcloud = WordCloud( max_words=50, background_color="white", stopwords = stopwords).generate(text)

    # Display the generated image:

    plt.figure(figsize=(10, 10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title("Foreign Policy Community: " +str(community),  fontdict={'fontsize': 25})

    plt.show()
#read house_legislation_116 from csv: bills

bills = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_legislation_116.csv', index_col = 0, parse_dates=["date_introduced"])



#converts values of relevant columns to lists of strings rather than strings

bills.cosponsors=bills.cosponsors.apply(literal_eval)

bills.subjects = bills.subjects.apply(literal_eval)

bills.committees = bills.committees.apply(literal_eval)

bills.related_bills = bills.related_bills.apply(literal_eval)



bills.head()
#read house_members_116 from csv: members

members = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_members_116.csv', index_col = 0)



#converts values of relevant columns to lists of strings rather than strings

members.committee_assignments = members.committee_assignments.apply(literal_eval)



members.head()
#define policy area and subjects by which to filter bills

policy_area = "International Affairs"

#subjects = set(['Sanctions', 'Conflict and wars', 'Terrorism', 'Foreign aid and international relief',  'International law and treaties', 'Alliances'])





#determine bills associated with the given policy area and subjects of interest

bills_of_interest = set()

for index, row in bills.iterrows():

    #if ((row.policy_area == policy_area) and bool(subjects.intersection(set(row.subjects)))):

    if ((row.policy_area == policy_area)):

        bills_of_interest.add(index)

        related = set(row.related_bills)

        bills_of_interest = bills_of_interest.union(related)



bills_of_interest = bills.reindex(bills_of_interest).dropna(how = 'all')



# MD: graph with directed edges from bill cosponsors to a bill sponsor

MD = nx.MultiDiGraph()



#add edges between sponsors and cosponsors for bills_of_interest

for index, row in bills_of_interest.iterrows():

    sponsor = [row.sponsor for i in range(len(row.cosponsors))]

    zipped = zip(row.cosponsors, sponsor)

    zipped = list(zipped)

    

    #set edge attribute related to bill policy_area

    MD.add_edges_from(zipped, bill = index, policy_area = row.policy_area, bill_progress = row.bill_progress)

    

#convert MD from a multi-directed graph to graph with weighted edges, with weights representing the number of edges between two nodes

G = nx.Graph()



for n, nbrs in MD.adjacency():

    for nbr, edict in nbrs.items():

        if (G.has_edge(n,nbr)) :

            G[n][nbr]['weight'] +=len(edict)

        else:

             G.add_edge(n, nbr, weight=len(edict))
# function for setting colors of nodes and edges

def get_paired_color_palette(size):

    palette = []

    for i in range(size*2):

        palette.append(plt.cm.Paired(i))

    return palette





#use louvain community detection algorithm to detect communities in G

communities =[]

louvain = community.best_partition(G, weight = 'weight', random_state = 42)

for i in set(louvain.values()):

    nodelist = [n for n in G.nodes if (louvain[n]==i)]

    communities.append(nodelist)



#sort communities by length of community

communities.sort(key = lambda x: len(x), reverse = True)



#make plot using matplotlib, networkx spring_layout, set_colors using cluster_count and get_paired_color_pallette

clusters_count = len(set(louvain.values()))

plt.figure(figsize=(10, 10))

light_colors = get_paired_color_palette(clusters_count)[0::2]

dark_colors = get_paired_color_palette(clusters_count)[1::2]

g = nx.drawing.layout.spring_layout(G, weight = 'weight', threshold = .0000000001)



for i in range(len(communities)):

    nodelist = communities[i]

    edgelist = [e for e in G.edges if (e[0] or e[1]) in nodelist]

    node_color = [light_colors[i] for _ in range(len(nodelist))]

    edge_color = [dark_colors[i] for _ in range(len(edgelist))]

    nx.draw_networkx_nodes(G, g, nodelist=nodelist, node_color=node_color, edgecolors='k', label = i)                                                                                                           

    nx.draw_networkx_edges(G, g, edgelist=edgelist, alpha=.5, edge_color=edge_color)



plt.title('Louvain clustering: International Affairs', fontdict={'fontsize': 25})

plt.legend()

plt.axis('off')

plt.show()

community_members = defaultdict()

community_bills = defaultdict()



for i in range(3):

    dic ={}

    index = []

    #set community_of_interest

    community_of_interest = i



    #create subgraph with nodes limited to the community_of_interest

    subgraph = MD.subgraph(communities[community_of_interest])



    #sort members of community_of_interest by in_degree centality

    community_df = pd.DataFrame.from_dict(nx.algorithms.centrality.in_degree_centrality(subgraph), orient = 'index', columns = ['centrality']).merge(members[['name','current_party', 'committee_assignments']], how = 'left', left_index = True, right_index = True).sort_values(by= 'centrality',ascending = False)

    community_members[i] = community_df

    

    # Tally bills in subgraph of community using a dictionary having keys of associated bills

    for u,v,d in subgraph.edges(data = True):

        if d['bill'] not in index:

            index.append(d['bill'])

        if d['bill'] not in dic:

            dic[d['bill']] = 1

        else:

            dic[d['bill']] += 1



    #create data frame from dic and sort by tally in descending order

    community_bills[i] = pd.DataFrame.from_dict(dic, orient = 'index',columns = ['tally']).sort_values(by = 'tally', ascending = False).merge(bills['title'],how = 'left', left_index =True, right_index= True)

    

l = []

index = []



for i in range(3):

    

    community_i  = i

    democrats = len(community_members[community_i].loc[community_members[community_i].current_party == 'Democratic'])

    republicans = len(community_members[community_i].loc[community_members[community_i].current_party == 'Republican'])

    independents = len(community_members[community_i].loc[community_members[community_i].current_party == 'Independent'])

    

    total_members = len(community_members[community_i])

    

    total_bills = len(community_bills[community_i])

    

    l.append([democrats, republicans, independents, total_members, total_bills])

    

    index.append('community_' +str(i))

    

    



        



df = pd.DataFrame(l, columns = ['democrats', 'republicans', 'independents', 'total_members', 'total_bills'],index = index)   

df.head()

    

    
pd.concat([community_members[0].head(10), community_members[1].head(10), community_members[2].head(10)], keys =

         ['community_0', 'community_1', 'community_2'])



pd.concat([community_bills[0].head(10), community_bills[1].head(10), community_bills[2].head(10)], keys =

         ['community_0', 'community_1', 'community_2'])







plot_wordcloud(bills, community_bills[0].index, '0')
plot_wordcloud(bills, community_bills[1].index, '1')
plot_wordcloud(bills, community_bills[2].index, '2')