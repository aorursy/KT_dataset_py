import pandas as pd

import networkx as nx

from node2vec import Node2Vec

from cytoolz import sliding_window

from collections import Counter



df = pd.read_csv("../input/titles.csv")

df.fillna("", inplace=True)
from nltk import wordpunct_tokenize

from nltk.corpus import stopwords

from string import punctuation

from cytoolz import isdistinct, topk, sliding_window, memoize

from operator import itemgetter

from itertools import product



stops = set(stopwords.words('english'))

stops.update(set(punctuation))



@memoize

def int_to_roman(x):

    """

    Normalizing titles like software engineer 3

    

    Also filters out numbers that are not likely part of a seniority description, i.e. 2000 

    """

    

    if not x.isnumeric():

        return x

    x = int(x)

    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)

    nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')

    result = []

    for i in range(len(ints)):

        count = int(x / ints[i])

        result.append(nums[i] * count)

        x -= ints[i] * count

    result = ''.join(result).lower()

    if any([n in result for n in ['M', 'C', 'D', 'X', 'L']]):

        return ""

    return ''.join(result).lower()
@memoize

def tokenize(x:str):

    return wordpunct_tokenize(x)



def lowercase(x:list):

    return [token.lower() for token in x]



def remove_stopwords(x:list, stopwords=stops):

    return [token for token in x if token not in stopwords]



def is_truthy(x):

    if x:

        return True

    return False



def preprocess_title(x:str):

    if pd.isna(x) or x == "":

        return []

    tokens = tokenize(x)

    tokens = lowercase(tokens)

    tokens = remove_stopwords(tokens)

    if not tokens:

        return []

    tokens = [int_to_roman(token) for token in tokens]

    tokens = list(filter(is_truthy, tokens))

    return tokens
# Get title counts



title_counts = Counter(df.stack().apply(lambda x: " ".join(preprocess_title(x))).values)

del title_counts['']
def get_gram_counts(tokens, best_grams):

    results = [] 

    seen = set()

    if not best_grams:

        gram_product = product(tokens, repeat=2)

    else:

        gram_product = product(tokens, best_grams)

    for title_grams in gram_product:

        if not isdistinct(title_grams):

            continue

        title = " ".join(title_grams)

        if title in seen:

            continue

        else:

            seen.add(title)

        count = title_counts.get(title, 0)

        results.append((title, count))

    return sorted(results, key=itemgetter(1), reverse=True)





@memoize

def optimize_title(x:str, topn=3, title_counts=title_counts):

    tokens = preprocess_title(x)

    if not tokens:

        return x

    if len(tokens)==1:

        return tokens[0]

    

    starting_score = title_counts.get(x, 1)

    best_ngrams = [(token, title_counts.get(token, 0)) for token in tokens]

    gram_counter = 2

    while gram_counter <= len(tokens): # Continue chaining tokens to get the highest score

        gram_counts = get_gram_counts(tokens, [token for token, score in best_ngrams])

        best_ngrams.extend(gram_counts)

        best_ngrams = list(topk(topn, best_ngrams, key=itemgetter(1)))

        if not any([g in best_ngrams for g in gram_counts]):  # The most recent get_gram_counts did not 'make the cut'

            best_ngram_found = topk(1, best_ngrams, key=itemgetter(1))[0]

            best_ngram, best_ngram_score = best_ngram_found

            if best_ngram_score > starting_score:

                return best_ngram

            else:

                return x

        gram_counter += 1

    return best_ngrams[0][0]



for title, count in sorted(title_counts.items(), key=itemgetter(1))[:50]:

    optimum_title = optimize_title(title)

    optimum_count = title_counts[optimum_title]

    print("{}, Count: {}  =======> {}, Count: {}".format(title, count, optimum_title, optimum_count))
%%time

g = nx.DiGraph()

title2optimum = {}

for i, row in df.iterrows():

    row_titles = [row[c] for c in df.columns]

    for title in row_titles:

        optimized_title = optimize_title(title)

        if title not in title2optimum:

            title2optimum[title] = optimized_title

    row_titles = [optimize_title(x) for x in row_titles]

    row_titles = list(filter(lambda x: x, row_titles))

    row_titles = reversed(row_titles)

    

    for prior_title, new_title in sliding_window(2, row_titles):

        if g.has_edge(prior_title, new_title):

                g.edges[(prior_title, new_title)]['weight'] += 1

        else:

            g.add_edge(prior_title, new_title, weight=1)
# What edges have a weight of 1?

weight_filter = lambda x: [(n1, n2) for n1, n2, weight in g.edges.data('weight') if weight ==x]

weight_filter_edges = weight_filter(1)

g.remove_edges_from(weight_filter_edges)  # Remove those edges



# What nodes are now unconnected?

degree_filter = lambda x: [node for node, degree in dict(g.degree()).items() if degree <= x]

degree_filter_nodes = degree_filter(0)

g.remove_nodes_from(degree_filter_nodes) # Remove those nodes



# We had an expectation that each job history should be at least 3 titles long. After pruning the graph we will likely have nodes that are now shorter than 3 and should be removed

two_degree_nodes = degree_filter(2)
two_degree_nodes
import operator

from operator import ge as greater_or_equal

from operator import le as less_or_equal



ug = g.to_undirected()



def filter_by_chain_length(node, chain_length, op, g=g):

    chain_counter = 0

    n_ancestors = len(nx.ancestors(g, node))

    chain_counter += n_ancestors

    if op == operator.ge:  # we may be able to skip a step

        if op(chain_counter, chain_length):

            return True

    

    n_descendants = len(nx.descendants(g, node))

    chain_counter += n_descendants

    return op(chain_counter, chain_length)
short_chain_nodes = list(filter(lambda x: filter_by_chain_length(x, 4, less_or_equal), two_degree_nodes))



print("Removing {} Short Chain Nodes".format(len(short_chain_nodes)))

g.remove_nodes_from(short_chain_nodes)
two_degree_nodes = [x for x in two_degree_nodes if x not in short_chain_nodes]

assert not list(filter(lambda x: filter_by_chain_length(x, 4, less_or_equal), two_degree_nodes))
# Making a temp folder for Joblib. Parallel execution can quickly eat up Kaggle's 16GB of memory



import os



temp_folder = r"/kaggle/working/temp_folder"



if not os.path.isdir(temp_folder):

    os.makedirs(temp_folder)

node2vec = Node2Vec(g, dimensions=64, walk_length=5, num_walks=200, workers=4, temp_folder=temp_folder, p=0.01)  # Use temp_folder for big graphs
model = node2vec.fit(min_count=2, window=3)
model.wv.save_word2vec_format('titles.wv')

model.save('titles_node2vec.model')

import pickle

with open('title2optimum.pkl', 'wb') as pfile:

    pickle.dump(title2optimum, pfile)
for title, _ in Counter(title_counts).most_common(100):

    if not g.edges(title):

        continue

    print(title.center(80, "="))

    sims = model.wv.most_similar(title)

    for sim_title, score in sims:

        print("\t{} : {}".format(sim_title, score))

    print("")