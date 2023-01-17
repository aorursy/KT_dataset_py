import numpy as np # linear algebra
import os # accessing directory structure
import json
import glob

#################### Custom functions to find norm of a matrix and perform some cosine distance similarity ############################
import numpy as np
def find_norm(syn0):
    syn0norm = (syn0 / np.sqrt((syn0 ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)
    return syn0norm


def argsort(x, topn=None, reverse=False):
    """
    Return indices of the `topn` smallest elements in array `x`, in ascending order.
    If reverse is True, return the greatest elements instead, in descending order.
    """
    x = np.asarray(x)  # unify code path for when `x` is not a numpy array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    # numpy >= 1.8 has a fast partial argsort, use that!
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order


def find_similar(des_norm, vec_norm):
    dists = np.dot(des_norm, vec_norm)

    best = argsort(dists, reverse=True)

    dist_sort = np.sort(dists)[::-1]

    return dist_sort, best


def similar(word, Vocab, idx2Vocab, MAT, topN=20):
    if word in Vocab:
        index = Vocab[word]
        query_vector = MAT[index]
        probs, indexes = find_similar(MAT, query_vector)

        words = [idx2Vocab[i] for i in indexes[:topN]]
        words_probs = dict(zip(words, probs[:topN]))

        return sorted(words_probs.items(), key=lambda x: x[1], reverse=True)
    else:
        return "OOV"


def similar_cross_matrix(query_vector , MAT , idx2Vocab , topN=20):

    probs, indexes = find_similar(MAT, query_vector)

    words = [idx2Vocab[i] for i in indexes[:topN]]
    words_probs = dict(zip(words, probs[:topN]))

    return sorted(words_probs.items(), key=lambda x: x[1], reverse=True)

print(glob.glob('../input/word2vec_bi_gram/word2vec_bi_gram/*'))
########################### Loading embeddings will take some time, as it is nearly 8 GB in size ##################################################
########################### https://github.com/s4sarath/gensim_ngram/blob/master/README.md
####### Note: it is advised to calcluate norm of the matrxix using find_norm(embeddings) for better results, but kernel is restarting due to memory issues, I guess, So skipping it
emebddings = np.load('../input/word2vec_bi_gram/word2vec_bi_gram/word2vec_bi_gram.syn0.npy') #### find_norm helps us to normalize embeddings, to simplify cosine distance to np.dot()
vocab      = json.load(open('../input/word2vec_bi_gram/word2vec_bi_gram/word2vec_bi_gram.vocab.json'))
vocab_reverse = {k:v for v, k in vocab.items()}
similar("brad pitt" , vocab , vocab_reverse , emebddings)
