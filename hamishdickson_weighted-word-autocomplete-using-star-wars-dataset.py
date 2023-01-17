# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import string
import pandas as pd

df_IV = pd.read_table("../input/SW_EpisodeIV.txt", error_bad_lines=False)
df_V = pd.read_table("../input/SW_EpisodeV.txt", error_bad_lines=False)
df_VI = pd.read_table("../input/SW_EpisodeVI.txt", error_bad_lines=False)
pd.set_option('display.max_colwidth', -1)
df_IV.columns = ['text']
df_V.columns = ['text']
df_VI.columns = ['text']

df_IV.head(5)
def prep_text(in_text):
    return in_text.split('"')[3:-1][0].lower().translate(str.maketrans("", "", string.punctuation)).split()
df_IV['clean_text'] = df_IV.apply(lambda row: prep_text(row['text']), axis=1)
df_V['clean_text'] = df_V.apply(lambda row: prep_text(row['text']), axis=1)
df_VI['clean_text'] = df_VI.apply(lambda row: prep_text(row['text']), axis=1)
df_IV.head(5)
df = pd.concat([df_IV, df_V, df_VI])
sentences = list()

for idx, row in df.iterrows():
    sentences.append(row['clean_text'])
sentences[:3]
flat_list = [item for sublist in sentences for item in sublist]
df_clean = pd.DataFrame(flat_list)
df_clean.columns = ['clean']
df_clean['lengths'] = df_clean.apply(lambda row: len(row['clean']), axis=1)
df_clean['lengths'].plot(kind='hist')
y = []

for item in df_clean['clean'].values:
    l = list(item)
    l.append("<eow>")
    y.append(l)
y[:5]
out_len = len(df_clean['clean'].unique())
out_len
class WeightedTrie:
    def __init__(self):
        self.count = 1
        self.tails = {}
def create_trie(in_text):
    trie = WeightedTrie()

    for word in y:
        curr = trie
        for letter in word:
            if letter in curr.tails:
                curr = curr.tails[letter]
                curr.count += 1
            else:
                new_trie = WeightedTrie()
                curr.tails[letter] = new_trie
                curr = new_trie
                
    return trie
trie = create_trie(y)
def predict(word_start, dic, n=3, width=26):
    cs = list(word_start)
    curr = dic
    # start working your way through the trie
    for c in cs:
        if c in curr.tails:
            curr = curr.tails[c]
        else:
            return "word not found, perhaps add it to known words?"
            
    # so at this point we're part way though the trie. In a lot of cases we haven't
    # finished fining the word we want - so let's do a BFS over the nodes at the next
    # layer, but only keep the most likely words to appear next (ie the words with
    # the highest count
    
    # basically BFS
    topn = []
    scores = []
    node_queue = []
    tmp = []
    
    for k, v in curr.tails.items():
        # there are a few different ways to score this. here we penalise longer words
        score = v.count / len(cs + [k])
        tmp.append((score, cs + [k], v.tails))
            
    # todo have a think about this - there might be a nicer way to do this
    # this takes O(W * N lg N) 
    tmp2 = sorted(tmp, reverse=True)
    
    for item in tmp2[:width]:
        node_queue.append(item)
    
    while node_queue and len(topn) < n:
        current_score, so_far, next_dic = node_queue.pop(0)

        if so_far[-1] == '<eow>':
            topn.append(so_far)
            scores.append(current_score)
    
        tmp = []
        for k, v in next_dic.items():
            score = (current_score + v.count) / len(so_far + [k])
            tmp.append((score, so_far + [k], v.tails))
        
        tmp2 = sorted(tmp, reverse=True)
        
        for item in tmp2[:width]:
            node_queue.append(item)

    return sorted([(s, ''.join(item[:-1])) for item, s in zip(topn, scores)], reverse=True)
%%time
predict("vade", trie)
%%time
predict("le", trie, n=10)
