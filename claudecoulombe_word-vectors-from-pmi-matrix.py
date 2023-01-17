from collections import Counter

import itertools



import nltk

from nltk.corpus import stopwords

import numpy as np

import pandas as pd

from scipy import sparse

from scipy.sparse import linalg 

from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv('../input/abcnews-date-text.csv')

df.head()
headlines = df['headline_text'].tolist()

# remove stopwords

stopwords_set = set(stopwords.words('english'))

headlines = [

    [tok for tok in headline.split() if tok not in stopwords_set] for headline in headlines

]

# remove single word headlines

headlines = [hl for hl in headlines if len(hl) > 1]

# show results

headlines[0:20]
tok2indx = dict()

unigram_counts = Counter()

for ii, headline in enumerate(headlines):

    if ii % 200000 == 0:

        print(f'finished {ii/len(headlines):.2%} of headlines')

    for token in headline:

        unigram_counts[token] += 1

        if token not in tok2indx:

            tok2indx[token] = len(tok2indx)

indx2tok = {indx:tok for tok,indx in tok2indx.items()}

print('done')

print('vocabulary size: {}'.format(len(unigram_counts)))

print('most common: {}'.format(unigram_counts.most_common(10)))
# note add dynammic window hyperparameter

back_window = 2

front_window = 2

skipgram_counts = Counter()

for iheadline, headline in enumerate(headlines):

    for ifw, fw in enumerate(headline):

        icw_min = max(0, ifw - back_window)

        icw_max = min(len(headline) - 1, ifw + front_window)

        icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]

        for icw in icws:

            skipgram = (headline[ifw], headline[icw])

            skipgram_counts[skipgram] += 1    

    if iheadline % 200000 == 0:

        print(f'finished {iheadline/len(headlines):.2%} of headlines')

        

print('done')

print('number of skipgrams: {}'.format(len(skipgram_counts)))

print('most common: {}'.format(skipgram_counts.most_common(10)))
row_indxs = []

col_indxs = []

dat_values = []

ii = 0

for (tok1, tok2), sg_count in skipgram_counts.items():

    ii += 1

    if ii % 1000000 == 0:

        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')

    tok1_indx = tok2indx[tok1]

    tok2_indx = tok2indx[tok2]

        

    row_indxs.append(tok1_indx)

    col_indxs.append(tok2_indx)

    dat_values.append(sg_count)

    

wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))

print('done')
def ww_sim(word, mat, topn=10):

    """Calculate topn most similar words to word"""

    indx = tok2indx[word]

    if isinstance(mat, sparse.csr_matrix):

        v1 = mat.getrow(indx)

    else:

        v1 = mat[indx:indx+1, :]

    sims = cosine_similarity(mat, v1).flatten()

    sindxs = np.argsort(-sims)

    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]

    return sim_word_scores



print('done')
ww_sim('strike', wwcnt_mat)
wwcnt_norm_mat = normalize(wwcnt_mat, norm='l2', axis=1)

print('done')
ww_sim('strike', wwcnt_norm_mat)
num_skipgrams = wwcnt_mat.sum()

assert(sum(skipgram_counts.values())==num_skipgrams)



# for creating sparse matrices

row_indxs = []

col_indxs = []



# pmi: pointwise mutual information

pmi_dat_values = []

# ppmi: positive pointwise mutual information

ppmi_dat_values = []

# spmi: smoothed pointwise mutual information

spmi_dat_values = []

# sppmi: smoothed positive pointwise mutual information

sppmi_dat_values = []



# Sum over words and contexts

sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()

sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()



# Smoothing

# According to [Levy, Goldberg & Dagan, 2015], the smoothing operation 

# should be done on the context 

alpha = 0.75

nca_denom = np.sum(sum_over_contexts**alpha)

# sum_over_words_alpha = sum_over_words**alpha

sum_over_contexts_alpha = sum_over_contexts**alpha



ii = 0

for (tok1, tok2), sg_count in skipgram_counts.items():

    ii += 1

    if ii % 1000000 == 0:

        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')

    tok1_indx = tok2indx[tok1]

    tok2_indx = tok2indx[tok2]

    

    nwc = sg_count

    Pwc = nwc / num_skipgrams



    nw = sum_over_contexts[tok1_indx]

    Pw = nw / num_skipgrams

    

    nc = sum_over_words[tok2_indx]

    Pc = nc / num_skipgrams

    

    pmi = np.log2(Pwc/(Pw*Pc))

    ppmi = max(pmi, 0)

    

#   nca = sum_over_words_alpha[tok2_indx]

    nca = sum_over_contexts_alpha[tok2_indx]

    Pca = nca / nca_denom



    spmi = np.log2(Pwc/(Pw*Pca))

    sppmi = max(spmi, 0)

    

    row_indxs.append(tok1_indx)

    col_indxs.append(tok2_indx)

    pmi_dat_values.append(pmi)

    ppmi_dat_values.append(ppmi)

    spmi_dat_values.append(spmi)

    sppmi_dat_values.append(sppmi)

        

pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))

ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))

spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))

sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))



print('done')
ww_sim('strike', pmi_mat)
ww_sim('strike', ppmi_mat)
ww_sim('strike', spmi_mat)
ww_sim('strike', sppmi_mat)
pmi_use = ppmi_mat

embedding_size = 50

uu, ss, vv = linalg.svds(pmi_use, embedding_size) 



print('done')
print('vocab size: {}'.format(len(unigram_counts)))

print('embedding size: {}'.format(embedding_size))

print('uu.shape: {}'.format(uu.shape))

print('ss.shape: {}'.format(ss.shape))

print('vv.shape: {}'.format(vv.shape))
unorm = uu / np.sqrt(np.sum(uu*uu, axis=1, keepdims=True))

vnorm = vv / np.sqrt(np.sum(vv*vv, axis=0, keepdims=True))

#word_vecs = unorm

#word_vecs = vnorm.T

word_vecs = uu + vv.T

word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs*word_vecs, axis=1, keepdims=True))



print('done')
def word_sim_report(word, sim_mat):

    sim_word_scores = ww_sim(word, word_vecs)

    for sim_word, sim_score in sim_word_scores:

        print(sim_word, sim_score)

        word_headlines = [hl for hl in headlines if sim_word in hl and word in hl][0:5]

        for headline in word_headlines:

            print(f'    {headline}')

            

print('done')
word = 'strike'

word_sim_report(word, word_vecs)
word = 'war'

word_sim_report(word, word_vecs)
word = 'bank'

word_sim_report(word, word_vecs)
word = 'car'

word_sim_report(word, word_vecs)

word = 'football'

word_sim_report(word, word_vecs)
word = 'tech'

word_sim_report(word, word_vecs)
# check a few things

alpha = 0.75

dsum = wwcnt_mat.sum()

nwc = skipgram_counts[(tok1, tok2)]

Pwc = nwc / dsum



indx1 = tok2indx[tok1]

indx2 = tok2indx[tok2]



nw = wwcnt_mat[indx1, :].sum()

Pw = nw / dsum

nc = wwcnt_mat[:, indx2].sum()

Pc = nc / dsum



nca = nc**alpha

nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten()**alpha)

Pca = nca / nca_denom



print('dsum=', dsum)

print('Pwc=', Pwc)

print('Pw=', Pw)

print('Pc=', Pc)

print('Pca=', Pca)

pmi1 = Pwc / (Pw * Pc)

pmi1a = Pwc / (Pw * Pca)

pmi2 = (nwc * dsum) / (nw * nc)



print('pmi1=', pmi1)

print('pmi1a=', pmi1a)

print('pmi2=', pmi2)