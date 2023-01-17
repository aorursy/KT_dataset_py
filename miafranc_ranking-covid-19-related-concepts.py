import nltk

import json

import codecs

import os

import regex

from gensim.models.callbacks import CallbackAny2Vec

import gensim.models

from pprint import pprint

import numpy as np

from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import dijkstra

import pickle

import string

import pandas as pd

from scipy.sparse.csgraph import connected_components

from scipy.spatial import distance_matrix



import matplotlib.pyplot as plt

from wordcloud import WordCloud

from pandas.plotting import parallel_coordinates

from scipy.spatial import voronoi_plot_2d, Voronoi

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection



import glob
plt.rcParams["figure.figsize"] = (20, 15)
root_path = '/kaggle/input/CORD-19-research-challenge'

all_json = glob.glob(f'{root_path}/biorxiv_medrxiv/**/*.json', recursive=True)

texts = []

for fname in all_json:

    f = codecs.open(fname, 'r', 'utf-8')

    js = json.load(f)

    f.close()



    texts.append(js['metadata']['title'] + ".")

    texts.extend([x['text'] for x in js['abstract']])

    texts.extend([x['text'] for x in js['body_text']])

#     print(texts[:10])
w1 = 'covid-19'
drugs_file = '/kaggle/input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv'

data = pd.read_csv(drugs_file)

concepts = [w.lower() for w in data.condition.unique() if isinstance(w, str)]  # medical conditions

# print(concepts[:10])
def tokenize(text, vocab_dict):

    """Tokenize text based on a dictionary (multiword, nested), always finding the longest matching

    in the dictionary (if it exists at all).

    """

    tokens = [w.lower() for w in nltk.word_tokenize(text)]

    tokens = list(filter(lambda token: token not in string.punctuation, tokens))  # remove punctuation characters

    tokens2 = []

    

    i = 0

    while i < len(tokens):

        t = []

        r = vocab_dict

        hashtag = False

        while i < len(tokens) and r.get(tokens[i], -1) != -1:

            t.append(tokens[i])

            r = r.get(tokens[i])

            if r.get('#', -1) != -1:

                t.append('#')

                hashtag = True

            i += 1

        

        if len(t) == 0 or hashtag == False: # if not in the dictionary at all: take the first token only, then continue from the second one

            i = i - len(t)

            tokens2.append(tokens[i])

            i += 1

        else: # if some prefix was found in the dict, take until the last # character, then continue from there

            j = len(t) - 1

            while j >= 0 and t[j] != '#': j -= 1

            tokens2.append("_".join(filter(lambda x: x != '#', t[:j])))

            i = i - (len(t) - j  - 1)

        

    return tokens2



def make_dict(vocab):

    """Make nested dictionary from multiword vocabulary.

    The # sign represents the end of a multiword token in the dict.  

    """

    vocab_dict = {}

    

    for w in vocab:

        tokens = regex.split(r'\s+', w)

        v = vocab_dict

        for t in tokens:

            if v.get(t, -1) == -1:

                v[t] = {}

            v = v[t]

        v['#'] = {}

        

    return vocab_dict



def make_simple_dict(vocab):

    """Make a dictionary by joining multiword term with the '_' character.

    """

    vocab_dict = {}

    

    for w in vocab:

        v = '_'.join(regex.split(r'\s+', w))

        vocab_dict[v] = 1

        

    return vocab_dict
vocab_dict = make_dict(concepts)  # nested dict for multi-word expressions

sentences = []  # training sequences for word2vec

for t in texts:

    t = regex.sub(r'\s+', ' ', t)

    sent = nltk.sent_tokenize(t)

    sentences.extend([tokenize(s, vocab_dict) for s in sent])

# print(sentences[:10])
class callback(CallbackAny2Vec):

    """Callback to print loss after each epoch

    """

    def __init__(self):

        self.epoch = 0



    def on_epoch_end(self, model):

        loss = model.get_latest_training_loss()

        if self.epoch == 0:

            print('Loss after epoch {}: {}'.format(self.epoch, loss))

        else:

            if self.epoch % 10 == 0:

                print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))

        self.epoch += 1

        self.loss_previous_step = loss



model = gensim.models.Word2Vec(sg=0, min_count=3, size=100, window=5)

model.build_vocab(sentences=sentences)

model.train(sentences=sentences, 

            total_examples=model.corpus_count,

            report_delay=1,

            epochs=100, 

            compute_loss=True,

            callbacks=[callback()])
wmap = {w:i for i, w in enumerate(model.wv.vocab)}



knn = 10  # k-nearest neighbors in the graph



row_ind = []

col_ind = []

data = []



it = 0

for w in model.wv.vocab:

    if it % 10000 == 0:

        print('Building graph: {} words done.'.format(it))

    it += 1



    sim_words = model.wv.similar_by_word(w, topn=knn)

    row_ind.extend([wmap[w]] * len(sim_words))

    col_ind.extend([wmap[sw[0]] for sw in sim_words])

    data.extend([1-sw[1] for sw in sim_words])



A = csr_matrix((data, (row_ind, col_ind)), shape=(len(wmap), len(wmap)), dtype=np.float64)
simple_dict = make_simple_dict(concepts)



min_d = dijkstra(A, directed=True, indices=[wmap[w1]])



graph_sims = []

for w2 in simple_dict.keys():

    if wmap.get(w2, -1) != -1:

        graph_sims.append((w2, min_d[0][wmap[w2]]))

graph_sims_orig = graph_sims
sims = []

for w2 in simple_dict.keys():

    if wmap.get(w2, -1) != -1:

        sims.append((w2, model.wv.similarity(w1, w2)))

sims_orig = sims
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))



graph_sims = [(x[0], 1./(1 + x[1])) for x in graph_sims]  # transforming distances to similarities

max_v = max([x[1] for x in graph_sims])

pprint(sorted(graph_sims, key=lambda x: x[1], reverse=True))

graph_sims = sorted(graph_sims, key=lambda x: x[1], reverse=False)  # because we want the 'best' result to be on top

ax1.barh(range(len(graph_sims)), [x[1] for x in graph_sims], align='center', color=[[0.9*(1 - x[1]/max_v)] * 3 for x in graph_sims])

plt.setp(ax1, yticks=range(len(graph_sims)), yticklabels=[x[0] for x in graph_sims])



wc = WordCloud(width=800, height=400, max_font_size=100, background_color='white', colormap='Oranges').generate_from_frequencies(dict(graph_sims))

ax2.imshow(wc)



plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))



sims = list(filter(lambda x: x[1] > 0 and x[1] != np.inf, sims))

max_v = max([x[1] for x in sims])

pprint(sorted(sims, key=lambda x: x[1], reverse=True))

sims = sorted(sims, key=lambda x: x[1], reverse=False)

ax1.barh(range(len(sims)), [x[1] for x in sims], align='center', color=[[0.9*(1 - x[1]/max_v)] * 3 for x in sims])

plt.setp(ax1, yticks=range(len(sims)), yticklabels=[x[0] for x in sims])



wc = WordCloud(width=800, height=400, max_font_size=100, background_color='white', colormap='Oranges').generate_from_frequencies(dict(sims))

ax2.imshow(wc)



plt.show()
res_graph = sorted(graph_sims_orig, key=lambda x: x[1], reverse=True)

ranks = {x[1][0]:x[0] for x in enumerate(res_graph)}

ranks1 = [ranks[x[0]] for x in res_graph]

res_w2v = sorted(sims_orig, key=lambda x: x[1])

ranks2 = [ranks[x[0]] for x in res_w2v]

data = [[ranks1[i], ranks2[i], i] for i in range(len(ranks))]

df = pd.DataFrame(data, columns =['graph', 'w2v', 'rank1'])



plt.figure(figsize=(20, 15))

ax = parallel_coordinates(df, 'rank1', colormap='Oranges', axvlines=False)

ax.legend().remove()

plt.yticks(range(len(res_graph)), [x[0] for x in res_graph])

plt.grid(b=None)

plt.show()
def isomap(D, n_components):

    N = D.shape[0]

    J = np.eye(N) - (1./N) * np.ones((N, N))

    u, s, vh = np.linalg.svd(-0.5 * np.matmul(J, np.matmul(np.power(D, 2), J)))

    print('(var. explained = {})'.format(np.sum(s[:n_components]) / np.sum(s)))

    s_k = np.power(np.diag(s[:n_components]), 0.5)

    return np.matmul(u[:, :n_components], s_k)



def get_subgraph(wmap, adj_matrix, simple_dict):

    words = []

    words_ind = []

    for w2 in simple_dict.keys():

        if wmap.get(w2, -1) != -1:

            words.append(w2)

            words_ind.append(wmap[w2])

    words.append(w1)

    words_ind.append(wmap[w1])



    min_d = dijkstra(adj_matrix, directed=True, indices=[words_ind])

    D = min_d[0,:,words_ind]

    

    cc = connected_components(D, directed=True, connection='strong')

    comp_ind = cc[1][-1]  # last one added is the central concept

    

    cc_ind = np.argwhere(cc[1] == comp_ind).T

    D = D[cc_ind[0],:][:,cc_ind[0]]

    words  = [words[i] for i in cc_ind[0]]

    

    return (D, words)



# From: https://gist.github.com/pv/8036995

def voronoi_finite_polygons_2d(vor, radius=None):

    """

    Reconstruct infinite voronoi regions in a 2D diagram to finite

    regions.

    Parameters

    ----------

    vor : Voronoi

        Input diagram

    radius : float, optional

        Distance to 'points at infinity'.

    Returns

    -------

    regions : list of tuples

        Indices of vertices in each revised Voronoi regions.

    vertices : list of tuples

        Coordinates for revised Voronoi vertices. Same as coordinates

        of input vertices, with 'points at infinity' appended to the

        end.

    """



    if vor.points.shape[1] != 2:

        raise ValueError("Requires 2D input")



    new_regions = []

    new_vertices = vor.vertices.tolist()



    center = vor.points.mean(axis=0)

    if radius is None:

        radius = vor.points.ptp().max()*2



    # Construct a map containing all ridges for a given point

    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):

        all_ridges.setdefault(p1, []).append((p2, v1, v2))

        all_ridges.setdefault(p2, []).append((p1, v1, v2))



    # Reconstruct infinite regions

    for p1, region in enumerate(vor.point_region):

        vertices = vor.regions[region]



        if all(v >= 0 for v in vertices):

            # finite region

            new_regions.append(vertices)

            continue



        # reconstruct a non-finite region

        ridges = all_ridges[p1]

        new_region = [v for v in vertices if v >= 0]



        for p2, v1, v2 in ridges:

            if v2 < 0:

                v1, v2 = v2, v1

            if v1 >= 0:

                # finite ridge: already in the region

                continue



            # Compute the missing endpoint of an infinite ridge



            t = vor.points[p2] - vor.points[p1] # tangent

            t /= np.linalg.norm(t)

            n = np.array([-t[1], t[0]])  # normal



            midpoint = vor.points[[p1, p2]].mean(axis=0)

            direction = np.sign(np.dot(midpoint - center, n)) * n

            far_point = vor.vertices[v2] + direction * radius



            new_region.append(len(new_vertices))

            new_vertices.append(far_point.tolist())



        # sort region counterclockwise

        vs = np.asarray([new_vertices[v] for v in new_region])

        c = vs.mean(axis=0)

        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])

        new_region = np.array(new_region)[np.argsort(angles)]



        # finish

        new_regions.append(new_region.tolist())



    return new_regions, np.asarray(new_vertices)
res = get_subgraph(wmap, A, simple_dict)

X = isomap(res[0], 2)

vor = Voronoi(X)

fig = voronoi_plot_2d(vor, line_colors='orange', line_width=4, line_alpha=0.6, point_size=0)



D = distance_matrix([X[-1,:]], X)[0]

max_d = np.max(D)



regions, vertices = voronoi_finite_polygons_2d(vor)



patches = []

for r in range(len(regions)):

    region = regions[r]

    polygon = [vertices[i] for i in regions[r]]

    plt.fill(*zip(*polygon), color=(np.sqrt(sum((X[-1]-vor.points[r])**2)) /max_d)*np.array([1.0, 0, 0]))

    patches.append(Polygon(polygon, visible=False, closed=True))



def onpick(event):

    ind = event.ind

    fig.axes[0].texts.clear()

    fig.axes[0].text(X[ind[0],0], X[ind[0],1], res[1][ind[0]], fontsize=12, color='white', fontweight='bold')

    fig.canvas.draw()

    print('onpick:', ind, res[1][ind[0]])

    

fig.canvas.mpl_connect('pick_event', onpick)

fig.axes[0].add_collection(PatchCollection(patches, alpha=0.0, picker=5))

plt.show()