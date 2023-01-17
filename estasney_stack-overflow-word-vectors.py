import re
import os
from itertools import combinations
from collections import Counter, defaultdict, OrderedDict
from math import log
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
import numpy as np
from gensim.utils import smart_open
import datetime

fp = "../input/posts_data.csv"

tag_search = re.compile(r"([^<>]+)")

for i, line in enumerate(smart_open(fp)):
    if i == 0:
        headers = line.decode().replace("\r\n", "").split(",")
    elif i > 0:
        break
class SmartCSV(object):
    
    def __init__(self, fp, headers):
        self.fp = fp
        self.headers = headers
        
    def __iter__(self):
        for i, line in enumerate(smart_open(self.fp)):
            if i == 0:
                continue
            line_to_dict = line.decode().replace("\r\n", "").split(",")
            line_to_dict = {k: v for k, v in zip(self.headers, line_to_dict)}
            yield line_to_dict
            
    def __getitem__(self, index):
        for i, line in enumerate(smart_open(self.fp)):
            if i == index:
                line_to_dict = line.decode().replace("\r\n", "").split(",")
                line_to_dict = {k: v for k, v in zip(self.headers, line_to_dict)}
                return line_to_dict
smart_csv = SmartCSV(fp, headers=headers)
class TagData(object):
    
    def __init__(self, smart_csv=smart_csv):
        self.smart_csv = smart_csv
        self.tag2id = defaultdict()
        self.tag2id.default_factory = self.tag2id.__len__
        self.id2tag = {}
        self.tag2created = {}
        self.tag_counts = Counter()
        self.date2idx = defaultdict()
        self.date2idx.default_factory = self.date2idx.__len__
        self.rows = []
        
    def fit(self):
        for line in self.smart_csv:
            line_tags_raw = line.get('Tags')
            if not line_tags_raw:
                continue
            line_tags = tag_search.findall(line_tags_raw)
            if not line_tags:
                continue
            # Creating id2tag
            for tag in line_tags:
                _ = self.tag2id[tag]
                
            tag_ids = list(filter(lambda x: x is not None, [self.tag2id[x] for x in line_tags]))
            
            # Counting occurences
            self.tag_counts.update(tag_ids)
            
            
            # Tracking new tags and their creation date
            post_date = datetime.datetime.strptime(line['CreationDate'].split("T")[0], "%Y-%m-%d")
            new_tags = list(filter(lambda x: x not in self.tag2created, tag_ids))
            for nt in new_tags:
                self.tag2created[nt] = post_date
                
            # Use date2idx to find which row to extend
            row_index = self.date2idx[post_date]
            
            try:
                _ = self.rows[row_index]
            except IndexError:
                self.rows.append([])
                
            self.rows[row_index].append(tag_ids)
            
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}
tag_data = TagData(smart_csv)
%time tag_data.fit()
%%time
# We have the number of times a tag appears available at tag_data.tag_counts

# We calculate the probability that a tag appears in a given post

# For each tag we divide its count by the number of tags that occur ON OR AFTER its created date

occurences = np.array(list(tag_data.tag_counts.values()))

# Working backwards from the last date...
segmented_counter = Counter()
segmented_dict = {}
segment_i = len(tag_data.rows) - 1
for day in reversed(tag_data.rows):
    for post in day:
        segmented_counter.update(post)
    segmented_dict[segment_i] = sum(segmented_counter.values())
    segment_i -= 1
# Building word_sums
word_sums = []

# For each tag we:
for tag in list(tag_data.tag_counts.keys()):
    # Get the tag created date
    created_date = tag_data.tag2created[tag]
    # Get the corresponding index for the creation date
    date_idx = tag_data.date2idx[created_date]
    # Lookup the accumulated counts from segemented_dict
    accum_counts = segmented_dict[date_idx]
    word_sums.append(accum_counts)

word_sums = np.array(word_sums)
probabilities = occurences / word_sums

del word_sums, occurences, segmented_dict, segmented_counter
cxp = {k: v for k, v in zip(list(tag_data.tag_counts.keys()), probabilities)}

del probabilities
import pandas as pd
df = pd.DataFrame.from_dict(list(cxp.items()))

import math
import matplotlib.pyplot as plt

%matplotlib inline

df[1] = df[1].apply(lambda x: math.log(x))
df[1].plot(kind='kde', figsize=(12, 8))

df = df.loc[df[1]>-15]
del cxp

cxp = {int(k): v for k, v in df.values}

del df
# we want to use cxp's keys and drop others from tag_data

tag_data.tag2id = dict(tag_data.tag2id)

for k in list(tag_data.id2tag.keys()):
    if k not in cxp:
        # get the tag2id before deleting
        tag_name = tag_data.id2tag[k]
        del tag_data.id2tag[k]
        del tag_data.tag2id[tag_name]
def stream_combo_tags(f=smart_csv, idx=tag_data.tag2id):
    for line in f:
        line_tags_raw = line.get('Tags')
        if not line_tags_raw:
            continue
        line_tags = tag_search.findall(line_tags_raw)
        line_tags = [idx.get(tag) for tag in line_tags]
        line_tags = list(filter(lambda x: x is not None, line_tags))
        if not line_tags or len(line_tags)<2:
            continue
        
        for x, y in set(map(tuple, map(sorted, combinations(line_tags, 2)))):
            yield x, y

sc = stream_combo_tags()
%time bigram_counts = Counter(sc)
# Perform a similar operation as above

# After summing a tag pair, we want to divide by the total # of occurences of bigrams ON OR AFTER the created date
# of the newer tag

# Working backwards from the last date...

segmented_counter_combos = Counter()
segmented_dict_combos = {}
segment_i_combos = len(tag_data.rows) - 1

for day in reversed(tag_data.rows):
    for post in day:
        post = list(filter(lambda x: x in tag_data.id2tag, post))
        post_combos = list(map(sorted, combinations(post, 2)))
        for combo in post_combos:
            segmented_counter_combos.update(combo)
    segmented_dict_combos[segment_i_combos] = sum(segmented_counter_combos.values())
    segment_i_combos -= 1
cxyp = {}

for bigram_pair, v in list(bigram_counts.items()):
    newest_tag_date = max([tag_data.tag2created[bigram_pair[0]], tag_data.tag2created[bigram_pair[1]]])
    # Get the row corresponding to date
    row_idx = tag_data.date2idx[newest_tag_date]
    # Get total counts from date on
    total_counts = segmented_dict_combos[row_idx]
    cxyp[bigram_pair] = v / total_counts
for x, y in map(sorted, combinations(['c', 'haskell', 'javascript', 'flask', 'python', 'django'], 2)):
    x_id, y_id = tag_data.tag2id[x], tag_data.tag2id[y]
    xy_prob = cxyp.get((x_id, y_id), 0)
    print("Probability of {} and {} occurring together : {:.4%}".format(x, y, xy_prob))  
# Note that the frequency of a tags occurence is directly correlated with its skipgram probability
# We want to normalize -> PMI
# Probability of skipgram (a, b) / probability(a) * probability (b)

for x, y in map(sorted, combinations(['c', 'haskell', 'clojure', 'sql', 'python', 'django'], 2)):
    x_id, y_id = sorted([tag_data.tag2id[x], tag_data.tag2id[y]])
    
    # We need 3 values - skipgram probability, x's probability and y's probability
    x_prob, y_prob, xy_prob = cxp[x_id], cxp[y_id], cxyp.get((x_id, y_id), 0)
    if xy_prob == 0:
        print("{} and {} did not occur together".format(x, y))
        continue
    pmi_value = log(xy_prob / (x_prob * y_prob))
    print("PMI of {} and {} : {:.8}".format(x, y, pmi_value))
pmi_samples = Counter()
data, rows, cols = [], [], []
for (x, y), n in cxyp.items():
    rows.append(x)
    cols.append(y)
    x_prob, y_prob, xy_prob = cxp[x_id], cxp[y_id], cxyp.get((x_id, y_id), 0)
    pmi_value = log(xy_prob / (x_prob * y_prob))
    data.append(pmi_value)
PMI = csc_matrix((data, (rows, cols)))
from sklearn.utils.extmath import svd_flip
# Copied From sklearn.decomposition.truncated_svd.fit_transform

U, Sigma, VT = svds(PMI, k=25)
# svds doesn't abide by scipy.linalg.svd/randomized_svd
# conventions, so reverse its outputs.
Sigma = Sigma[::-1]
U, VT = svd_flip(U[:, ::-1], VT[::-1])
U *= Sigma

# Note: sklearn does not return normalized 
norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
U /= norms

for x in ['python', 'java', 'flask', 'clojure', 'sql', 'sqlalchemy', 'javascript']:
    dd = np.dot(U, U[tag_data.tag2id[x]]) # Cosine similarity for this unigram against all others.
    sims = np.argsort(-1 * dd)[:25]
    readable = [(tag_data.id2tag[n], "{:n}".format(dd[n]), tag_data.tag_counts[n]) for n in sims if tag_data.tag2id[x] != n]
    print("{} : {}".format(x, tag_data.tag_counts[tag_data.tag2id[x]]))
    print("-" * 10)
    s = ''
    for n in readable:
        s += "({}, {}, {}), ".format(n[0], n[1], n[2])
    print(s)
    print()
    print("=" * 80)
    print()
# Copied From sklearn.decomposition.truncated_svd.fit_transform

U, Sigma, VT = svds(PMI, k=100)
# svds doesn't abide by scipy.linalg.svd/randomized_svd
# conventions, so reverse its outputs.
Sigma = Sigma[::-1]
U, VT = svd_flip(U[:, ::-1], VT[::-1])
U *= Sigma

# Note: sklearn does not return normalized 
norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
U /= norms
for x in ['python', 'java', 'flask', 'clojure', 'sql', 'sqlalchemy', 'javascript']:
    dd = np.dot(U, U[tag_data.tag2id[x]]) # Cosine similarity for this unigram against all others.
    sims = np.argsort(-1 * dd)[:25]
    readable = [(tag_data.id2tag[n], "{:n}".format(dd[n]), tag_data.tag_counts[n]) for n in sims if tag_data.tag2id[x] != n]
    print("{} : {}".format(x, tag_data.tag_counts[tag_data.tag2id[x]]))
    print("-" * 10)
    s = ''
    for n in readable:
        s += "({}, {}, {}), ".format(n[0], n[1], n[2])
    print(s)
    print()
    print("=" * 80)
    print()