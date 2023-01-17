from __future__ import print_function, division
from collections import Counter
from itertools import combinations
from math import log, floor
from pprint import pformat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from string import punctuation
from time import time
import numpy as np
import pandas as pd
import string as strin
print('Ready')
# 1. Read and preprocess titles from HN posts.
punctrans = str.maketrans(dict.fromkeys(punctuation))
def tokenize(title):
    x = title.lower() # Lowercase
    x = x.encode('ascii', 'ignore').decode() # Keep only ascii chars.
    x = x.translate(punctrans) # Remove punctuation
    return x.split() # Return tokenized.

t0 = time()
df = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv', usecols=['title'])
texts_tokenized = df['title'].apply(tokenize)
print('%.3lf seconds (%.5lf / iter)' % (time() - t0, (time() - t0) / len(df)))
# 2a. Compute unigram and bigram counts.
# A unigram is a single word (x). A bigram is a pair of words (x,y).
# Bigrams are counted for any two terms occurring in the same title.
# For example, the title "Foo bar baz" has unigrams [foo, bar, baz]
# and bigrams [(bar, foo), (bar, baz), (baz, foo)]
t0 = time()
cx = Counter()
cxy = Counter()
for text in texts_tokenized:
    for x in text:
        cx[x] += 1
    for x, y in map(sorted, combinations(text, 2)):
        cxy[(x, y)] += 1
print('%.3lf seconds (%.5lf / iter)' %
      (time() - t0, (time() - t0) / len(texts_tokenized)))
# 2b. Remove frequent and infrequent unigrams.
# Pick arbitrary occurrence count thresholds to eliminate unigrams occurring
# very frequently or infrequently. This decreases the vocab size substantially.
print('%d tokens before' % len(cx))
t0 = time()
min_count = (1 / 1000) * len(df)
max_count = (1 / 50) * len(df)
for x in list(cx.keys()):
    if cx[x] < min_count or cx[x] > max_count:
        del cx[x]
print('%.3lf seconds (%.5lf / iter)' % (time() - t0, (time() - t0) / len(cx)))
print('%d tokens after' % len(cx))
print('Most common:', cx.most_common()[:25])
# 2c. Remove frequent and infrequent bigrams.
# Any bigram containing a unigram that was removed must now be removed.
t0 = time()
for x, y in list(cxy.keys()):
    if x not in cx or y not in cx:
        del cxy[(x, y)]
print('%.3lf seconds (%.5lf / iter)' % (time() - t0, (time() - t0) / len(cxy)))
# 3. Build unigram <-> index lookup.
t0 = time()
x2i, i2x = {}, {}
for i, x in enumerate(cx.keys()):
    x2i[x] = i
    i2x[i] = x
print('%.3lf seconds (%.5lf / iter)' % (time() - t0, (time() - t0) / len(cx)))
# 4. Sum unigram and bigram counts for computing probabilities.
# i.e. p(x) = count(x) / sum(all counts).
t0 = time()
sx = sum(cx.values())
sxy = sum(cxy.values())
print('%.3lf seconds (%.5lf / iter)' %
      (time() - t0, (time() - t0) / (len(cx) + len(cxy))))
# 5. Accumulate data, rows, and cols to build sparse PMI matrix
# Recall from the blog post that the PMI value for a bigram with tokens (x, y) is: 
# PMI(x,y) = log(p(x,y) / p(x) / p(y)) = log(p(x,y) / (p(x) * p(y)))
# The probabilities are computed on the fly using the sums from above.
t0 = time()
pmi_samples = Counter()
data, rows, cols = [], [], []
for (x, y), n in cxy.items():
    rows.append(x2i[x])
    cols.append(x2i[y])
    data.append(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))
    pmi_samples[(x, y)] = data[-1]
PMI = csc_matrix((data, (rows, cols)))
print('%.3lf seconds (%.5lf / iter)' % (time() - t0, (time() - t0) / len(cxy)))
print('%d non-zero elements' % PMI.count_nonzero())
print('Sample PMI values\n', pformat(pmi_samples.most_common()[:10]))
# 6. Factorize the PMI matrix using sparse SVD aka "learn the unigram/word vectors".
# This part replaces the stochastic gradient descent used by Word2vec
# and other related neural network formulations. We pick an arbitrary vector size k=20.
t0 = time()
U, _, _ = svds(PMI, k=20)
print('%.3lf seconds' % (time() - t0))
# 7. Normalize the vectors to enable computing cosine similarity in next cell.
# If confused see: https://en.wikipedia.org/wiki/Cosine_similarity#Definition
t0 = time()
norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
U /= np.maximum(norms, 1e-7)
print('%.3lf seconds' % (time() - t0))
# 8. Show some nearest neighbor samples as a sanity-check.
# The format is <unigram> <count>: (<neighbor unigram>, <similarity>), ...
# From this we can see that the relationships make sense.
k = 5
for x in ['facebook', 'twitter', 'instagram', 'messenger', 'hack', 'security', 
          'deep', 'encryption', 'cli', 'venture', 'paris']:
    dd = np.dot(U, U[x2i[x]]) # Cosine similarity for this unigram against all others.
    s = ''
    # Compile the list of nearest neighbor descriptions.
    # Argpartition is faster than argsort and meets our needs.
    for i in np.argpartition(-1 * dd, k + 1)[:k + 1]:
        if i2x[i] == x: continue
        xy = tuple(sorted((x, i2x[i])))
        s += '(%s, %.3lf) ' % (i2x[i], dd[i])
    print('%s, %d\n %s' % (x, cx[x], s))
    print('-' * 10)
# 9. Measure the alphabetical bias between similar words, i.e. check the average alphabetical
# distance between a word and its closest neighbors. Similarity should be quite independent
# of positions in the alphabet, so this value should not be too low.

def alphabetical_distance(word_a, word_b):
    # For simplicity, just the distance between first characters
    return abs(strin.ascii_lowercase.index(word_a[0]) - strin.ascii_lowercase.index(word_b[0]))
    
n_neighbours = 10

average_distances = []

for word in cx.keys():
    dd = np.dot(U, U[x2i[word]]) 
    distances = []
    
    for i in np.argpartition(-1 * dd, n_neighbours + 1)[:n_neighbours + 1]:
        if i2x[i] == word: continue
        
        try:
            distances.append(alphabetical_distance(word, i2x[i]))
        except ValueError:
            pass  # Happens when first character of either `word` or `i2x[i]` is numerical
        
    try:
        average_distances.append(sum(distances) / float(len(distances)))
    except ZeroDivisionError:
        pass  # Happens when first character of `word` is numerical
    
print("Average alphabetical distance: {}".format(sum(average_distances) / float(len(average_distances))))
print("Histogram:")
print(sorted(Counter([floor(x) for x in average_distances]).items()))

