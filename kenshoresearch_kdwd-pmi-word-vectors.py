from collections import Counter

import json

import os

import random

import re



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm



from scipy import sparse

from scipy.sparse import linalg

from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import cosine_distances

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import euclidean_distances
sns.set()

sns.set_context('talk')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
NUM_KLAT_LINES = 5_343_564

MIN_UNIGRAM_COUNT = 100

WINDOW = 5

MAX_PAGES = 100_000

INTROS_ONLY = True

kdwd_path = os.path.join("/kaggle", "input", "kensho-derived-wikimedia-data")
def tokenizer(text):

    return text.strip().lower().split()
class KdwdLinkAnnotatedText:

    

    def __init__(self, file_path, intros_only=False, max_pages=1_000_000_000):

        self._file_path = file_path

        self._intros_only = intros_only

        self._max_pages = max_pages

        self._num_lines = NUM_KLAT_LINES

        self.pages_to_parse = min(self._num_lines, self._max_pages)

        

    def __iter__(self):

        with open(self._file_path) as fp:

            for ii, line in enumerate(fp):

                page = json.loads(line)

                for section in page['sections']:

                    yield section['text']

                    if self._intros_only:

                        break

                if ii + 1 >= self.pages_to_parse:

                    break
file_path = os.path.join(kdwd_path, "link_annotated_text.jsonl")

klat_intros_2 = KdwdLinkAnnotatedText(file_path, intros_only=True, max_pages=2)

klat_intros = KdwdLinkAnnotatedText(file_path, intros_only=INTROS_ONLY, max_pages=MAX_PAGES)
two_intros = [intro for intro in klat_intros_2]
two_intros
def filter_unigrams(unigrams, min_unigram_count):

    tokens_to_drop = [

        token for token, count in unigrams.items() 

        if count < min_unigram_count]                                                                                 

    for token in tokens_to_drop:                                                             

        del unigrams[token]

    return unigrams
def get_unigrams(klat):

    unigrams = Counter()

    for text in tqdm(

        klat, total=klat.pages_to_parse, desc='calculating unigrams'

    ):

        tokens = tokenizer(text)

        unigrams.update(tokens)

    return unigrams
unigrams = get_unigrams(klat_intros)

print("token count: {}".format(sum(unigrams.values())))                          

print("vocabulary size: {}".format(len(unigrams)))
unigrams = filter_unigrams(unigrams, MIN_UNIGRAM_COUNT)

print("token count: {}".format(sum(unigrams.values())))                          

print("vocabulary size: {}".format(len(unigrams))) 
tok2indx = {tok: indx for indx, tok in enumerate(unigrams.keys())}

indx2tok = {indx: tok for tok, indx in tok2indx.items()}
def get_skipgrams(klat, max_window, tok2indx, seed=938476):

    rnd = random.Random()

    rnd.seed(a=seed)

    skipgrams = Counter()

    for text in tqdm(

        klat, total=klat.pages_to_parse, desc='calculating skipgrams'

    ):

        

        tokens = tokenizer(text)

        vocab_indices = [tok2indx[tok] for tok in tokens if tok in tok2indx]

        num_tokens = len(vocab_indices)

        if num_tokens == 1:

            continue

        for ii_word, word in enumerate(vocab_indices):

            

            window = rnd.randint(1, max_window)

            ii_context_min = max(0, ii_word - window)

            ii_context_max = min(num_tokens - 1, ii_word + window)

            ii_contexts = [

                ii for ii in range(ii_context_min, ii_context_max + 1) 

                if ii != ii_word]

            for ii_context in ii_contexts:

                context = vocab_indices[ii_context]

                skipgram = (word, context)

                skipgrams[skipgram] += 1 



    return skipgrams
skipgrams = get_skipgrams(klat_intros, WINDOW, tok2indx)

print("number of unique skipgrams: {}".format(len(skipgrams)))

print("number of skipgrams: {}".format(sum(skipgrams.values())))

most_common = [

    (indx2tok[sg[0][0]], indx2tok[sg[0][1]], sg[1]) 

    for sg in skipgrams.most_common(25)]

print('most common: {}'.format(most_common))
def get_count_matrix(skipgrams, tok2indx):

    row_indxs = []                                                                       

    col_indxs = []                                                                       

    dat_values = []                                                                      

    for skipgram in tqdm(

        skipgrams.items(), 

        total=len(skipgrams), 

        desc='building count matrix row,col,dat'

    ):

        (tok_word_indx, tok_context_indx), sg_count = skipgram

        row_indxs.append(tok_word_indx)

        col_indxs.append(tok_context_indx)

        dat_values.append(sg_count)

    print('building sparse count matrix')

    return sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
count_matrix = get_count_matrix(skipgrams, tok2indx)
# normalize rows

count_matrix_l2 = normalize(count_matrix, norm='l2', axis=1)
# demonstrate normalization

irow=10

row = count_matrix_l2.getrow(irow).toarray().flatten()

print(np.sqrt((row*row).sum()))



row = count_matrix.getrow(irow).toarray().flatten()

print(np.sqrt((row*row).sum()))
xx1 = count_matrix.data

xx2 = count_matrix_l2.data

nbins = 30



fig, axes = plt.subplots(1, 2, figsize=(18,8))



ax = axes[0]

counts, bins, patches = ax.hist(xx1, bins=nbins, density=True, log=True)

ax.set_xlabel('count_matrix')

ax.set_ylabel('fraction')



ax = axes[1]

counts, bins, patches = ax.hist(xx2, bins=nbins, density=True, log=True)

ax.set_xlabel('count_matrix_l2')

ax.set_xlim(-0.05, 1.05)

ax.set_ylim(1e-4, 5e1)



fig.suptitle('Distribution of Embedding Matrix Values');
def ww_sim(word, mat, tok2indx, topn=10):

    """Calculate topn most similar words to word"""

    indx = tok2indx[word]

    if isinstance(mat, sparse.csr_matrix):

        v1 = mat.getrow(indx)

    else:

        v1 = mat[indx:indx+1, :]

    sims = cosine_similarity(mat, v1).flatten()

    #dists = cosine_distances(mat, v1).flatten()

    dists = euclidean_distances(mat, v1).flatten()

    sindxs = np.argsort(-sims)

    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]

    return sim_word_scores
word = 'city'
ww_sim(word, count_matrix, tok2indx)
ww_sim(word, count_matrix_l2, tok2indx)
word = "the"

context = "of"

word_indx = tok2indx[word]

context_indx = tok2indx[context]

print('pound_wc for ({},{}) from skipgrams: {}'.format(

    word, context, skipgrams[(word_indx, context_indx)]))

print('pound_wc for ({},{}) from count_matrix: {}'.format(

    word, context, count_matrix[word_indx, context_indx]))
sum_over_words = np.array(count_matrix.sum(axis=0)).flatten()    # sum over rows

sum_over_contexts = np.array(count_matrix.sum(axis=1)).flatten() # sum over columns



pound_w_check1 = count_matrix.getrow(word_indx).sum()

pound_w_check2 = sum_over_contexts[word_indx]

print('pound_w for "{}" from getrow then sum: {}'.format(word, pound_w_check1))

print('pound_w for "{}" from sum_over_contexts: {}'.format(word, pound_w_check2))



pound_c_check1 = count_matrix.getcol(context_indx).sum()

pound_c_check2 = sum_over_words[context_indx]

print('pound_c for "{}" from getcol then sum: {}'.format(context, pound_c_check1))

print('pound_c for "{}" from sum_over_words: {}'.format(context, pound_c_check2))
def get_ppmi_matrix(skipgrams, count_matrix, tok2indx, alpha=0.75):

    

    # for standard PPMI

    DD = sum(skipgrams.values())

    sum_over_contexts = np.array(count_matrix.sum(axis=1)).flatten()

    sum_over_words = np.array(count_matrix.sum(axis=0)).flatten()

        

    # for context distribution smoothing (cds)

    sum_over_words_alpha = sum_over_words**alpha

    Pc_alpha_denom = np.sum(sum_over_words_alpha)

        

    row_indxs = []

    col_indxs = []

    ppmi_dat_values = []   # positive pointwise mutual information

    

    for skipgram in tqdm(

        skipgrams.items(), 

        total=len(skipgrams), 

        desc='building ppmi matrix row,col,dat'

    ):

        

        (tok_word_indx, tok_context_indx), pound_wc = skipgram

        pound_w = sum_over_contexts[tok_word_indx]

        pound_c = sum_over_words[tok_context_indx]

        pound_c_alpha = sum_over_words_alpha[tok_context_indx]



        Pwc = pound_wc / DD

        Pw = pound_w / DD

        Pc = pound_c / DD

        Pc_alpha = pound_c_alpha / Pc_alpha_denom



        pmi = np.log2(Pwc / (Pw * Pc_alpha))

        ppmi = max(pmi, 0)

        

        row_indxs.append(tok_word_indx)

        col_indxs.append(tok_context_indx)

        ppmi_dat_values.append(ppmi)



    print('building ppmi matrix')    

    return sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
ppmi_matrix = get_ppmi_matrix(skipgrams, count_matrix, tok2indx)
word = 'city'

ww_sim(word, ppmi_matrix, tok2indx)
embedding_size = 200

uu, ss, vv = linalg.svds(ppmi_matrix, embedding_size)
print('vocab size: {}'.format(len(unigrams)))

print('ppmi size: {}'.format(ppmi_matrix.shape))

print('embedding size: {}'.format(embedding_size))

print('uu.shape: {}'.format(uu.shape))

print('ss.shape: {}'.format(ss.shape))

print('vv.shape: {}'.format(vv.shape))
# Dont do this for full run or we'll run out of RAM



#x = (uu.dot(np.diag(ss)).dot(vv))[word_indx, :]

#y = (uu.dot(np.diag(ss)).dot(vv))[context_indx, :]

#print((x * y).sum())



#x = (uu.dot(np.diag(ss)))[word_indx, :]

#y = (uu.dot(np.diag(ss)))[context_indx, :]

#print((x * y).sum())
p = 0.5

svd_word_vecs = uu.dot(np.diag(ss**p))

print(svd_word_vecs.shape)
nbins = 20

fig, axes = plt.subplots(2, 2, figsize=(16,14), sharey=False)



ax = axes[0,0]

xx = count_matrix.data

ax.hist(xx, bins=nbins, density=True, log=True)

ax.set_xlabel('word_counts')

ax.set_ylabel('fraction')



ax = axes[0,1]

xx = count_matrix_l2.data

ax.hist(xx, bins=nbins, density=True, log=True)

ax.set_xlim(-0.05, 1.05)

ax.set_xlabel('word_counts_l2')



ax = axes[1,0]

xx = ppmi_matrix.data

ax.hist(xx, bins=nbins, density=True, log=True)

ax.set_xlabel('PPMI')

ax.set_ylabel('fraction')



ax = axes[1,1]

xx = svd_word_vecs.flatten()

ax.hist(xx, bins=nbins, density=True, log=True)

ax.set_xlabel('SVD(p=0.5)-PPMI')



fig.suptitle('Distribution of Embedding Matrix Values');
word = 'car'

sims = ww_sim(word, svd_word_vecs, tok2indx)

for sim in sims:

    print('  ', sim)
word = 'king'

sims = ww_sim(word, svd_word_vecs, tok2indx)

for sim in sims:

    print('  ', sim)
word = 'queen'

sims = ww_sim(word, svd_word_vecs, tok2indx)

for sim in sims:

    print('  ', sim)
word = 'news'

sims = ww_sim(word, svd_word_vecs, tok2indx)

for sim in sims:

    print('  ', sim)
word = 'hot'

sims = ww_sim(word, svd_word_vecs, tok2indx)

for sim in sims:

    print('  ', sim)
svd_word_vecs.shape
from sklearn.manifold import TSNE
svd_2d = TSNE(n_components=2, random_state=3847).fit_transform(svd_word_vecs)
svd_2d
word='city'

size = 3

indx = tok2indx[word]

cen_vec = svd_2d[indx,:]

dxdy = np.abs(svd_2d - cen_vec) 

bmask = (dxdy[:,0] < size) & (dxdy[:,1] < size)

sub = svd_2d[bmask]





fig, ax = plt.subplots(figsize=(15,15))

ax.scatter(sub[:,0], sub[:,1])

ax.set_xlim(cen_vec[0] - size, cen_vec[0] + size)

ax.set_ylim(cen_vec[1] - size, cen_vec[1] + size)

for ii in range(len(indx2tok)):

    if not bmask[ii]:

        continue

    plt.annotate(

        indx2tok[ii],

        xy=(svd_2d[ii,0], svd_2d[ii,1]),

        xytext=(5, 2),

        textcoords='offset points',

        ha='right',

        va='bottom')