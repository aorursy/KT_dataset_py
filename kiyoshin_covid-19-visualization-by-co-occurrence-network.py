# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#setting

word_asso_all = 20000

word_asso = 2000

n_word_lower = 30

edge_threshold = 0.01
import re

# df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

df.shape
df.columns
#clean...

title = df.copy()

title = title.dropna(subset=['title'])

title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)

title['title'] = title['title'].str.lower()

title['title'] = title['title'].str.replace('and ', ' ', regex=True)

title['title'] = title['title'].str.replace('a ', ' ', regex=True)

title['title'] = title['title'].str.replace('of ', ' ', regex=True)

title['title'] = title['title'].str.replace('t ', ' ', regex=True)

title['title'] = title['title'].str.replace('h ', ' ', regex=True)

title['title'] = title['title'].str.replace('is ', ' ', regex=True)

title['title'] = title['title'].str.replace('the ', ' ', regex=True)

title['title'] = title['title'].str.replace('for ', ' ', regex=True)

title['title'] = title['title'].str.replace('in ', ' ', regex=True)

title['title'] = title['title'].str.replace('n ', ' ', regex=True)

title['title'] = title['title'].str.replace('are ', ' ', regex=True)

title['title'] = title['title'].str.replace('to ', ' ', regex=True)

title['title'] = title['title'].str.replace('cd ', ' ', regex=True)

title['title'] = title['title'].str.replace('g ', ' ', regex=True)

title['title'] = title['title'].str.replace('m ', ' ', regex=True)

title['title'] = title['title'].str.replace('e ', ' ', regex=True)

title['title'] = title['title'].str.replace('o ', ' ', regex=True)
#Why title ? over RAM

title['key_vaccine'] = title['title'].str.find('vaccine') 

title.head()
included_vaccine = title.loc[title['key_vaccine'] != -1]

included_vaccine
from pathlib import Path

import itertools

import collections

import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', 10)



data_dir_path = Path('.').joinpath('data')

result_dir_path = Path('.').joinpath('result')



if not result_dir_path.exists():

    result_dir_path.mkdir(parents=True)



lines =  included_vaccine['title']



sentences = [line.replace('\n', '').split(' ') for line in lines if not ('Heading' in line)]

#  1 word del....

sentences = [sentence for sentence in sentences if (len(sentence) >= 2)]

sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]

sentence_combinations = [[tuple(sorted(words)) for words in sentence] for sentence in sentence_combinations]

print('Word combinations..\n')

for combinations in sentence_combinations[:3]:

    print(combinations)
#  combination　1dimension

target_combinations = []

for sentence in sentence_combinations:

    target_combinations.extend(sentence)



##------------------------------------  Jaccard 

# Jaccard



combi_count = collections.Counter(target_combinations)



#  word combination

word_associates = []

for key, value in combi_count.items():

    word_associates.append([key[0], key[1], value])



word_associates = pd.DataFrame(word_associates, columns=['word1', 'word2', 'intersection_count'])



target_words = []

for word in target_combinations:

    target_words.extend(word)



word_count = collections.Counter(target_words)

word_count = [[key, value] for key, value in word_count.items()]

word_count = pd.DataFrame(word_count, columns=['word', 'count'])



word_associates = pd.merge(word_associates, word_count, left_on='word1', right_on='word', how='left')

word_associates.drop(columns=['word'], inplace=True)

word_associates.rename(columns={'count': 'count1'}, inplace=True)

word_associates = pd.merge(word_associates, word_count, left_on='word2', right_on='word', how='left')

word_associates.drop(columns=['word'], inplace=True)

word_associates.rename(columns={'count': 'count2'}, inplace=True)



word_associates['union_count'] = word_associates['count1'] + word_associates['count2'] - word_associates['intersection_count']

word_associates['jaccard_coefficient'] = word_associates['intersection_count'] / word_associates['union_count']



print('Jaccard :')

print(word_associates.head())
jaccard_coefficients = word_associates['jaccard_coefficient']

group_numbers = []

for coefficient in jaccard_coefficients:

    if coefficient < 0.0003:

        group_numbers.append(0)

    elif coefficient < 0.0006:

        group_numbers.append(1)

    elif coefficient < 0.0009:

        group_numbers.append(2)

    elif coefficient < 0.0012:

        group_numbers.append(3)

    else:

        group_numbers.append(4)

#     if coefficient < 0.003:

#         group_numbers.append(0)

#     elif coefficient < 0.006:

#         group_numbers.append(1)

#     elif coefficient < 0.009:

#         group_numbers.append(2)

#     elif coefficient < 0.012:

#         group_numbers.append(3)

#     else:

#         group_numbers.append(4) 

        

        

        

        

word_associates['group_number'] = group_numbers



word_associates_group_sum = word_associates.groupby('group_number').count()

word_associates_group_sum.reset_index(inplace=True)

print(word_associates_group_sum.loc[:, ['group_number', 'word1']])

print('')



sns.pairplot(hue='group_number', data=word_associates.sample(200).loc[:, ['count1', 'count2', 'group_number']])




word_associates.query('count1 >= @n_word_lower & count2 >= @n_word_lower', inplace=True)

word_associates.rename(columns={'word1':'node1', 'word2':'node2', 'jaccard_coefficient':'value'}, inplace=True)



# plot_network(data=word_associates, edge_threshold=edge_threshold)



# Co-occurrence network

def plot_network(data, edge_threshold=0., fig_size=(15, 15), file_name=None, dir_path=None):



    nodes = list(set(data['node1'].tolist()+data['node2'].tolist()))



    G = nx.Graph()



    G.add_nodes_from(nodes)



    #  add

    #  edge_threshold tree weight

    for i in range(len(data)):

        row_data = data.iloc[i]

        if row_data['value'] > edge_threshold:

            G.add_edge(row_data['node1'], row_data['node2'], weight=row_data['value'])



    # 1 node del

    isolated = [n for n in G.nodes if len([i for i in nx.all_neighbors(G, n)]) == 0]

    for n in isolated:

        G.remove_node(n)



    plt.figure(figsize=fig_size)

    pos = nx.spring_layout(G, k=0.3)  # k = node



    pr = nx.pagerank(G)



    # node size

    nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()),

                           cmap=plt.cm.Reds,

                           alpha=0.7,

                           node_size=[60000*v for v in pr.values()])

    # fonts

    nx.draw_networkx_labels(G, pos, fontsize=18, font_family='IPAexGothic', font_weight="bold")



    # edge

    edge_width = [d["weight"] * 100 for (u, v, d) in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="darkgrey", width=edge_width)



    plt.axis('off')



    if file_name is not None:

        if dir_path is None:

            dir_path = Path('.').joinpath('image')

        if not dir_path.exists():

            dir_path.mkdir(parents=True)

        plt.savefig(dir_path.joinpath(file_name), bbox_inches="tight")
plot_network(data=word_associates.head(word_asso_all), edge_threshold=edge_threshold)
plot_network(data=word_associates.head(word_asso), edge_threshold=edge_threshold)