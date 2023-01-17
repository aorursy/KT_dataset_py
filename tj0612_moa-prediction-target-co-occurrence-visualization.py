# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path

import itertools

import collections

import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv",index_col="sig_id")

df.head()
target_sum = df.sum(axis=0)

target_sum.hist(bins=50)

target_sum.sort_values(ascending=False)[:10]
for col in df.columns:

    df[col] = df[col].replace(1,col)

sentences=df.iloc[:,:].replace(0,np.nan).stack().groupby(level=0).apply(list).tolist()



sentences = [sentence for sentence in sentences if len(sentence) > 1]

sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]

sentence_combinations = [[tuple(sorted(words)) for words in sentence] for sentence in sentence_combinations]

target_combinations = []

for sentence in sentence_combinations:

    target_combinations.extend(sentence)



combi_count = collections.Counter(target_combinations)



word_associates = []

for key, value in combi_count.items():

    word_associates.append([key[0], key[1], value])



word_associates = pd.DataFrame(word_associates, columns=['word1', 'word2', 'intersection_count'])



target_words = []

for word in target_combinations:

    target_words.extend(word)



word_count = pd.DataFrame((df!=0).sum(axis=0),columns=["count"])

word_count["word"] = word_count.index



word_associates = pd.merge(word_associates, word_count, left_on='word1', right_on='word', how='left')

word_associates.drop(columns=['word'], inplace=True)

word_associates.rename(columns={'count': 'count1'}, inplace=True)

word_associates = pd.merge(word_associates, word_count, left_on='word2', right_on='word', how='left')

word_associates.drop(columns=['word'], inplace=True)

word_associates.rename(columns={'count': 'count2'}, inplace=True)



# calc Dice coefficient

word_associates['union_count'] = word_associates['count1'] + word_associates['count2'] - word_associates['intersection_count']

word_associates['dice_coefficient'] = word_associates['intersection_count'] * 2 / (word_associates['count1'] + word_associates["count2"])



print('Get Dice coefficient')

print(word_associates.head())
word_associates.dice_coefficient.hist(bins=100)
word_associates=word_associates.sort_values(by="dice_coefficient",ascending=False)

word_associates[word_associates.dice_coefficient>=0.6]
dice_coefficients = word_associates['dice_coefficient']

group_numbers = []

for coefficient in dice_coefficients:

    if coefficient < 0.025:

        group_numbers.append(0)

    elif coefficient < 0.04:

        group_numbers.append(1)

    elif coefficient < 0.08:

        group_numbers.append(2)

    elif coefficient < 0.15:

        group_numbers.append(3)

    else:

        group_numbers.append(4)

word_associates['group_number'] = group_numbers



word_associates_group_sum = word_associates.groupby('group_number').count()

word_associates_group_sum.reset_index(inplace=True)

print(word_associates_group_sum.loc[:, ['group_number', 'word1']])

print('')

#word_associates.loc[:, ['count1', 'count2', 'group_number']]

word_associates.group_number.value_counts()

sns.pairplot(hue='group_number', data=word_associates.loc[:, ['count1', 'count2', 'group_number']])

#plt.savefig(image_dir_path.joinpath(base_file_name+'_jaccard_group_plot.png'))
def plot_network(data, edge_threshold=0., n_word_lower = 10, fig_size=(30, 20)):

    data = data.query('count1 >= @n_word_lower & count2 >= @n_word_lower')

    data = data.rename(columns={'word1':'node1', 'word2':'node2', 'dice_coefficient':'value'})

    nodes = list(set(data['node1'].tolist()+data['node2'].tolist()))



    G = nx.Graph()

    G.add_nodes_from(nodes)



    for i in range(len(data)):

        row_data = data.iloc[i]

        if row_data['value'] > edge_threshold:

            G.add_edge(row_data['node1'], row_data['node2'], weight=row_data['value'])



    isolated = [n for n in G.nodes if len([i for i in nx.all_neighbors(G, n)]) == 0]

    for n in isolated:

        G.remove_node(n)



    plt.figure(figsize=fig_size)

    pos = nx.spring_layout(G, k=0.6)



    pr = nx.pagerank(G)



    nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()),

                           cmap=plt.cm.Reds,

                           alpha=0.8,

                           node_size=[60000*v for v in pr.values()])



    nx.draw_networkx_labels(G, pos, fontsize=12, font_weight="bold")

    

    edge_width = [d["weight"] * 30 for (u, v, d) in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="grey", width=edge_width)

    plt.axis('off')
plot_network(data=word_associates, edge_threshold=0.04, n_word_lower=20)