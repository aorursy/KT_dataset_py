import os
import sys
import pandas as pd
import nltk as nl
nl.download('punkt')
nl.download('stopwords')
nl.download('wordnet')
nl.download('averaged_perceptron_tagger')
# I do see kaggle.com complain about the above downloads, but perhaps it is a firewall issue..

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
stoplist = set(nl.corpus.stopwords.words('english'))
ltz = WordNetLemmatizer()
def calc_list_olap(lw1, lw2):
    """Given two lists of words, calculate the overlapping fraction: 
       1. Overlap in smallest list. 2. Overlap in the combined list. """
    s1 = set(lw1)    # duplicate words are subsumed, obviously..
    s2 = set(lw2)
    lop = len(s1 & s2)
    lunion = len(s1 | s2)
    lmin = min(len(s1), len(s2))
    folap = round(lop/lmin, 1) if lmin > 0 else  0       # this is basically the best overlap ratio in either of the two sentences.
    folapu = round(lop/lunion, 1) if lunion > 0 else  0  # this is the overlap accounting for both sentences.
    return (folap, folapu)
# https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk  -- list of tags
NounTags = ('N', 'NP', 'NN', 'NNS', 'NNP', 'NNPS')
AdjTags = ('JJ', 'JJR', 'JJS')
AdvTags = ('RB', 'RBR', 'RP')
VbTags = ('VB', 'VBD', 'VBG', 'VBN', 'VBZ')

def brk_sentence(sentence):
    """Break a sentence into its Nouns, Adjectives, Adverbs, Verbs list & all tokens """
    tkns = nl.word_tokenize(sentence.lower())
    tagl = nl.pos_tag(tkns)
    nw = [(ltz.lemmatize(word), clas) for (word, clas) in tagl if word not in stoplist]
    nouns = [word for (word, clas) in nw if clas in NounTags]
    adjs = [word for (word, clas) in nw if clas in AdjTags]
    advs = [word for (word, clas) in nw if clas in AdvTags]
    vbs = [word for (word, clas) in nw if clas in VbTags]
    return (nouns, adjs, advs, vbs, tkns)
def sentences_overlap(q1, q2):
    """Given a sentence pair, check the overlap in it, its Nouns, Adjectives, Adverbs and Verbs."""
    try:
        (nouns1, adj1, adv1, vb1, tkn1) = brk_sentence(q1)     
        (nouns2, adj2, adv2, vb2, tkn2) = brk_sentence(q2)

        ra, rtu = calc_list_olap(tkn1, tkn2)  # ratio of overlap on all tokens of two sentences.
        rn, rnu = calc_list_olap(nouns1, nouns2) # .. overlap on nouns..
        rv, rvu = calc_list_olap(vb1, vb2) # .. overlap on verbs..
        rj, rjv = calc_list_olap(adj1, adj2) # .. overlap on adjectives..
        rd, rdv = calc_list_olap(adv1, adv2) # .. overlap on adverbs.. 
        return (ra, rn, rv, rj, rd)

    except:
        e = sys.exc_info()
        print(e)
        print(q1, q2)
        return (-1, -1, -1, -1, -1)  # .. under rare exception: seen with a couple of rows for NaN cases.
print("Reading train.csv ..")
df = pd.read_csv('../input/train.csv')
df.head()
N = len(df)
N1 = round(N/2)   # N1 marks the boundary index. First half of the csv data is used for training. second half for verification..
print("Training dataset is first", N1,  "rows -- ", round(sum(df[0:N1]['is_duplicate']) / N1 * 100, 1), "% duplicate-questions sets")
print("Verification dataset is the last", (N-N1),  "rows -- ", round(sum(df[N1:]['is_duplicate']) / (N-N1) * 100, 1), "% duplicate-questions sets")
print("Processing grammar words for Training set. This can take some time...")
trx = pd.DataFrame()   # training 
trx['r_all'],trx['r_n'],trx['r_v'],trx['r_adj'],trx['r_adv'] = zip(*df[0:N1].apply(lambda x: sentences_overlap(x['question1'], x['question2']), axis=1))
trx.describe()    # training set.
print("Processing grammar words for Verification set. This can take some time...")
vrx = pd.DataFrame()   # verification
vrx['r_all'],vrx['r_n'],vrx['r_v'],vrx['r_adj'],vrx['r_adv'] = zip(*df[N1:].apply(lambda x: sentences_overlap(x['question1'], x['question2']), axis=1))
vrx.describe()    # verification set.
for col in ['r_all', 'r_n', 'r_v', 'r_adj', 'r_adv']:
    print("Correlation between 'is_duplicate' and", "'" + col + "'", ':', round(pearsonr(df['is_duplicate'][0:N1], trx[col])[0], 2))
lr = LogisticRegression()
lrmodl = lr.fit(trx.values, df['is_duplicate'][0:N1].values)
yp = lr.predict(vrx.values)
matching = df['is_duplicate'][N1:] == yp
print("Logistic Regression model accuracy:", round(sum(matching) / (N - N1) * 100, 1), "%")