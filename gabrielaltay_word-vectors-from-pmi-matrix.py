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
unigram_counts = Counter()

for ii, headline in enumerate(headlines):

    if ii % 200000 == 0:

        print(f'finished {ii/len(headlines):.2%} of headlines')

    for token in headline:

        unigram_counts[token] += 1



tok2indx = {tok: indx for indx, tok in enumerate(unigram_counts.keys())}

indx2tok = {indx: tok for tok,indx in tok2indx.items()}

print('done')

print('vocabulary size: {}'.format(len(unigram_counts)))

print('most common: {}'.format(unigram_counts.most_common(10)))
# note add dynammic window hyperparameter



# note we store the token vocab indices in the skipgram counter

# note we use Levy, Goldberg, Dagan notation (word, context) as opposed to (focus, context)

back_window = 2

front_window = 2

skipgram_counts = Counter()

for iheadline, headline in enumerate(headlines):

    tokens = [tok2indx[tok] for tok in headline]

    for ii_word, word in enumerate(tokens):

        ii_context_min = max(0, ii_word - back_window)

        ii_context_max = min(len(headline) - 1, ii_word + front_window)

        ii_contexts = [

            ii for ii in range(ii_context_min, ii_context_max + 1) 

            if ii != ii_word]

        for ii_context in ii_contexts:

            skipgram = (tokens[ii_word], tokens[ii_context])

            skipgram_counts[skipgram] += 1    

    if iheadline % 200000 == 0:

        print(f'finished {iheadline/len(headlines):.2%} of headlines')

        

print('done')

print('number of skipgrams: {}'.format(len(skipgram_counts)))

most_common = [

    (indx2tok[sg[0][0]], indx2tok[sg[0][1]], sg[1]) 

    for sg in skipgram_counts.most_common(10)]

print('most common: {}'.format(most_common))
row_indxs = []

col_indxs = []

dat_values = []

ii = 0

for (tok1, tok2), sg_count in skipgram_counts.items():

    ii += 1

    if ii % 1000000 == 0:

        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')    

    row_indxs.append(tok1)

    col_indxs.append(tok2)

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

    
ww_sim('strike', wwcnt_mat)
# normalize each row using L2 norm

wwcnt_norm_mat = normalize(wwcnt_mat, norm='l2', axis=1)
# demonstrate normalization

row = wwcnt_mat.getrow(10).toarray().flatten()

print(np.sqrt((row*row).sum()))



row = wwcnt_norm_mat.getrow(10).toarray().flatten()

print(np.sqrt((row*row).sum()))
ww_sim('strike', wwcnt_norm_mat)
num_skipgrams = wwcnt_mat.sum()

assert(sum(skipgram_counts.values())==num_skipgrams)



# for creating sparce matrices

row_indxs = []

col_indxs = []



pmi_dat_values = []    # pointwise mutual information

ppmi_dat_values = []   # positive pointwise mutial information

spmi_dat_values = []   # smoothed pointwise mutual information

sppmi_dat_values = []  # smoothed positive pointwise mutual information



# reusable quantities



# sum_over_rows[ii] = sum_over_words[ii] = wwcnt_mat.getcol(ii).sum()

sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()

# sum_over_cols[ii] = sum_over_contexts[ii] = wwcnt_mat.getrow(ii).sum()

sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()



# smoothing

alpha = 0.75

sum_over_words_alpha = sum_over_words**alpha

nca_denom = np.sum(sum_over_words_alpha)







ii = 0

for (tok_word, tok_context), sg_count in skipgram_counts.items():



    ii += 1

    if ii % 1000000 == 0:

        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')



    # here we have the following correspondance with Levy, Goldberg, Dagan

    #========================================================================

    #   num_skipgrams = |D|

    #   nwc = sg_count = #(w,c)

    #   Pwc = nwc / num_skipgrams = #(w,c) / |D|

    #   nw = sum_over_cols[tok_word]    = sum_over_contexts[tok_word] = #(w)

    #   Pw = nw / num_skipgrams = #(w) / |D|

    #   nc = sum_over_rows[tok_context] = sum_over_words[tok_context] = #(c)

    #   Pc = nc / num_skipgrams = #(c) / |D|

    #

    #   nca = sum_over_rows[tok_context]^alpha = sum_over_words[tok_context]^alpha = #(c)^alpha

    #   nca_denom = sum_{tok_content}( sum_over_words[tok_content]^alpha )

    

    nwc = sg_count

    Pwc = nwc / num_skipgrams

    nw = sum_over_contexts[tok_word]

    Pw = nw / num_skipgrams

    nc = sum_over_words[tok_context]

    Pc = nc / num_skipgrams

    

    nca = sum_over_words_alpha[tok_context]

    Pca = nca / nca_denom

    

    # note 

    # pmi = log {#(w,c) |D| / [#(w) #(c)]} 

    #     = log {nwc * num_skipgrams / [nw nc]}

    #     = log {P(w,c) / [P(w) P(c)]} 

    #     = log {Pwc / [Pw Pc]}

    pmi = np.log2(Pwc/(Pw*Pc))   

    ppmi = max(pmi, 0)

    spmi = np.log2(Pwc/(Pw*Pca))

    sppmi = max(spmi, 0)

    

    row_indxs.append(tok_word)

    col_indxs.append(tok_context)

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
def word_sim_report(word, sim_mat):

    sim_word_scores = ww_sim(word, word_vecs)

    for sim_word, sim_score in sim_word_scores:

        print(sim_word, sim_score)

        word_headlines = [hl for hl in headlines if sim_word in hl and word in hl][0:5]

        for headline in word_headlines:

            print(f'    {headline}')
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