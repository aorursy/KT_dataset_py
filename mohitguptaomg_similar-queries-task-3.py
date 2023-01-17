import numpy as np

import pandas as pd

import networkx as nx

from matplotlib import pyplot as plt

import warnings



TRAIN_PATH = "../input/train.csv"

tr = pd.read_csv(TRAIN_PATH)

pos = tr[tr.is_duplicate==1]
tr.info()
pos.info()
g = nx.Graph()

g.add_nodes_from(pos.question1)

g.add_nodes_from(pos.question2)

edges = list(pos[['question1', 'question2']].to_records(index=False))

g.add_edges_from(edges)
len(pos), g.number_of_edges()
cc = filter(lambda x : (len(x) > 3) and (len(x) < 10), 

            nx.connected_component_subgraphs(g))

g1 = next(cc)

g1.nodes()


with warnings.catch_warnings():

    warnings.simplefilter('ignore')

    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)

    plt.show()
g1 = next(cc)

g1.nodes()
with warnings.catch_warnings():

    warnings.simplefilter('ignore')

    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)

    plt.show()
g1 = next(cc)

g1.nodes()
with warnings.catch_warnings():

    warnings.simplefilter('ignore')

    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)

    plt.show()
cc = nx.connected_component_subgraphs(g)

node_cts = list(sub.number_of_nodes() for sub in cc)

cc = nx.connected_component_subgraphs(g)

edge_cts = list(sub.number_of_edges() for sub in cc)
cts = pd.DataFrame({'nodes': node_cts, 'edges': edge_cts})
cts[:10]
cts['mean_deg'] = 2 * cts.edges / cts.nodes

cts[:10]
cts.nodes.clip_upper(10).value_counts().sort_index()
cts.plot.scatter('nodes', 'edges')

plt.show()
df = pd.read_csv("../input/train.csv")

df.head()

import gensim.utils
df.shape
def read_questions(row,column_name):

    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))

    
documents = []

for index, row in df.iterrows():

    documents.append(read_questions(row,"question1"))

    if row["is_duplicate"] == 0:

        documents.append(read_questions(row,"question2"))
model = gensim.models.Word2Vec(size=150, window=10, min_count=2, sg=1, workers=10)

model.build_vocab(documents)  # prepare the model vocabulary
model.train(sentences=documents, total_examples=len(documents), epochs=model.iter)
word_vectors = model.wv

count = 0

for word in word_vectors.vocab:

    if count<10:

        print(word)

        count += 1

    else:

        break
len(word_vectors.vocab)
vector = model.wv["immigration"]  # numpy vector of a word

len(vector)
wanted_words = []

count = 0

for word in word_vectors.vocab:

    if count<500:

        wanted_words.append(word)

        count += 1

    else:

        break

wanted_vocab = dict((k, word_vectors.vocab[k]) for k in wanted_words if k in word_vectors.vocab)

from sklearn.manifold import TSNE
X = model[wanted_vocab] # X is an array of word vectors, each vector containing 150 tokens

tsne_model = TSNE(perplexity=40, n_components=3, init="pca", n_iter=5000, random_state=23)

Y = tsne_model.fit_transform(X)
#Plot the t-SNE output

fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(Y[:, 0], Y[:, 1])

words = list(wanted_vocab)

for i, word in enumerate(words):

    plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))

ax.set_yticklabels([]) #Hide ticks

ax.set_xticklabels([]) #Hide ticks

_ = plt.show()
w1 = "phone"

model.wv.most_similar(positive=w1, topn=5)
model.wv.doesnt_match(["government","corruption","peace"])