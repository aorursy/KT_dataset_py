import numpy as np
from scipy.linalg import svd, inv
import re, random
import itertools
import os
print(os.listdir("../input"))
with open('../input/negative','r') as fopen:
    tweets = fopen.read().split('\n')
with open('../input/positive','r') as fopen:
    tweets += fopen.read().split('\n')
len(tweets)
tweets = [i.lower() for i in tweets]

### remove urls
tweets = [i.replace('http\S+|www.\S+', '') for i in tweets]

### remove emoji's
def filter_emoji(in_str):
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', in_str)

def remove_repeating_chars(in_str):
    return ''.join(''.join(s)[:2] for _, s in itertools.groupby(in_str))

tweets = [filter_emoji(i) for i in tweets]
tweets = [i.replace('@[\w\-]+:?', '') for i in tweets]
tweets = [i.replace('[\"\']+', '') for i in tweets]
tweets = [remove_repeating_chars(i) for i in tweets]
class LSA:
    def __init__(self, corpus, tfidf=False):
        self.corpus = corpus
        self.vocabulary = list(set(' '.join(self.corpus).split()))
        if tfidf:
            self._tfidf()
        else:
            self._bow()
        self._calc_svd()
        
    def _calc_svd(self):
        self.U, self.S, self.Vt = svd(self.tfidf.T, full_matrices =False)
        
    def _bow(self):
        self.tfidf = np.zeros((len(self.corpus),len(self.vocabulary)))
        for no, i in enumerate(self.corpus):
            for text in i.split():
                self.tfidf[no, self.vocabulary.index(text)] += 1
    
    def _tfidf(self):
        idf = {}
        for i in self.vocabulary:
            idf[i] = 0
            for k in self.corpus:
                if i in k.split():
                    idf[i] += 1
            idf[i] = np.log(idf[i] / len(self.corpus))
        self.tfidf = np.zeros((len(self.corpus),len(self.vocabulary)))
        for no, i in enumerate(self.corpus):
            for text in i.split():
                self.tfidf[no, self.vocabulary.index(text)] += 1
            for text in i.split():
                self.tfidf[no, self.vocabulary.index(text)] = self.tfidf[no, self.vocabulary.index(text)] * idf[text]
def find_sentences(keyword, corpus):
    d = []
    for content in [i for i in corpus if i.find(keyword)>=0]:
        a = content.split()
        d.append(a)
    return ' '.join([j for i in d for j in i if re.match("^[a-zA-Z_-]*$", j) and len(j) > 1])

def compare(string1, string2, corpus, tfidf=False):
    queries = [find_sentences(string1, corpus), find_sentences(string2, corpus)]
    lsa = LSA(queries,tfidf=tfidf)
    Vt = lsa.Vt
    S = np.diag(lsa.S)
    vectors =[(np.dot(S,Vt[:,0]), np.dot(S,Vt[:,i])) for i in range(len(Vt))]
    angles = [np.arccos(np.dot(a,b) / (np.linalg.norm(a,2)* np.linalg.norm(b,2))) for a,b in vectors[1:]]
    return np.abs(1 - float(angles[0])/float(np.pi/2))
compare('tv3', 'kerajaan', tweets)
compare('najib', 'kerajaan', tweets)
compare('tv3', 'najib', tweets)
compare('bn', 'kerajaan', tweets)
compare('umno', 'kerajaan', tweets)
compare('umno', 'bn', tweets)
compare('mahathir', 'pakatan', tweets)
