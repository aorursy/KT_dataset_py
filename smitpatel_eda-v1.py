# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import brown
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
brown_vocab = brown.words(categories=brown.categories())
len(brown_vocab)
brown_vocab_lower = list(map(lambda x:x.lower(),brown_vocab))
brown_vocab_lower = set(brown_vocab_lower)
# pronouns
fp = ['I','we','me', 'us', 'my', 'our', 'mine', 'ours'] # singular & plural
sp = ['you' ,'your' ,'yours'] # singular & plural
tps=['he', 'she','it','him','her','his', 'hers','its'] # singular
tpp = ['they','them','their','theirs'] # plural
intensive_pr = ['myself','yourself','herself','himself','itself','ourselves','yourselves','themselves'] # singular & plural
interrogative = ['what','whatever','which','whichever','who','whoever','whom','whomever','whose']
neg = ["cant","couldn't","shan't","shouldn't","wouldn't","haven't","didn't","not","never","won't"]
# ('|').join(fp)
# ('|').join(sp)
# ('|').join(tps)
# ('|').join(tpp)
# ('|').join(interrogative)
# ('|').join(intensive_pr) 
doc = "hello-hello you shouldn't gove,98 a fuck dam about those bastards."
(" ").join(re.findall(r"[a-zA-Z0-9\']+", doc))

def get_features(doc):
    
    i1 = len(re.findall(r'(?=\bI\b|\bwe\b|\bme\b|\bus\b|\bmy\b|\bour\b|\bmine\b|\bours\b)', doc)) # FP
    i2 = len(re.findall(r'(?=\byou\b|\byour\b|\byours\b)', doc)) # sP
    i3 = len(re.findall(r'(?=\bhe\b|\bshe\b|\bit\b|\bhim\b|\bher\b|\bhis\b|\bhers\b|\bits\b)', doc)) # tps
    i4 = len(re.findall(r'(?=\bthey\b|\bthem\b|\btheir\b|\btheirs\b)', doc)) # tpp
    i5 = len(re.findall(r'(?=\bwhat\b|\bwhatever\b|\bwhich\b|\bwhichever\b|\bwho\b|\bwhoever\b|\bwhom\b|\bwhomever\b|\bwhoses\b)', doc)) # interrogative
    i6 = len(re.findall(r'(?=\bmyself\b|\byourself\b|\bherself\b|\bhimself\b|\bitself\b|\bourselves\b|\byourselves\b|\bthemselves\b)', doc)) # intensive
    i7 = len(re.findall(r'(?=\bfor\b|\band\b|\bnor\b|\bbut\b|\byet\b|\bso\b|\bbefore\b|\bonce\b|\bsince\b|\bthough\b|\bwhile\b|\bas\b|\bbecause\b|\bafter\b)', doc)) # conjunctions
    
    i8 = len(doc) - len( re.findall('[a-zA-Z]', doc)) - doc.count(' ') - len(re.findall('[0-9]', doc))
    i9 = doc.count(',')
    i10 = doc.count('?')
    i11 = len(doc.split())
    # negs
    i12 = len(re.findall(r'(?=\bcan\'t\b|\bcouldn\'t\b|\bshan\'t\b|\bshouldn\'t\b|\bwouldn\'t\b|\bhaven\'t\b|\bdidn\'t\b|\bnot\b|\bnever\b|\bwon\'t\b|\bdon\'t\b|\bhadn\'t\b|\bcant\b|\bcouldnt\b|\bshant\b|\bshouldnt\b|\bwouldnt\b|\bhavent\b|\bdidnt\b|\bwont\b|\bdont\b|\bhadnt\b)', doc))
    
#     spelling mistakes
    temp_doc = (" ").join(re.findall(r"[a-zA-Z0-9\']+", doc))
    i13 = len(set(temp_doc.split())-brown_vocab_lower)
    
    return (i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13)
tqdm.pandas()
df['feat_vect'] = df['question_text'].progress_apply(get_features)
df['fp'] = df['feat_vect'].apply(lambda x: x[0])
df['sp'] = df['feat_vect'].apply(lambda x: x[1])
df['tps'] = df['feat_vect'].apply(lambda x: x[2])
df['tpp'] = df['feat_vect'].apply(lambda x: x[3])
df['interrogative'] = df['feat_vect'].apply(lambda x: x[4])
df['intensive'] = df['feat_vect'].apply(lambda x: x[5])
df['conjunction'] = df['feat_vect'].apply(lambda x: x[6])
df['special_chars'] = df['feat_vect'].apply(lambda x: x[7])
df['commas'] = df['feat_vect'].apply(lambda x: x[8])
df['qm'] = df['feat_vect'].apply(lambda x: x[9])
df['len'] = df['feat_vect'].apply(lambda x: x[10])
df['negs'] = df['feat_vect'].apply(lambda x: x[11])
df['sm'] = df['feat_vect'].apply(lambda x: x[12])
df.head()
sincere = df[df['target']==0]
insincere = df[df['target']==1]
def get_stats_df(class_df):
    columns = ['fp', 'sp', 'tps', 'tpp', 'interrogative', 'intensive', 'conjunction', 'commas', 'qm',
               'special_chars', 'len','negs','sm']
    stats_dict = dict()
    for each_col in columns:
        col_name = each_col+'_prob'
        temp = class_df.groupby(each_col).count()
        temp[col_name] = temp['qid']/temp['qid'].sum()
        temp.sort_values(by=col_name, ascending=False, inplace=True)
        stats_dict[each_col] = temp[['qid',col_name]]
        
    return stats_dict
sincere_stats = get_stats_df(sincere)
insincere_stats = get_stats_df(insincere)
def get_plot(feat_name):
    
    col_name = feat_name+'_prob'
    fig = plt.figure(figsize=(5,4), dpi=100)
    ax1 = plt.axes()
    ax1.set_title(feat_name)
    ax1.scatter(x=sincere_stats[feat_name].index, y =sincere_stats[feat_name][col_name], s=10, c='b', marker="x", label='sincere')
    ax1.scatter(x=insincere_stats[feat_name].index, y =insincere_stats[feat_name][col_name], s=10, c='r', marker="o", label='insincere')
    ax1.set_xlabel('count_' + feat_name, fontsize=10)
    ax1.set_ylabel('prob', fontsize=10)
    plt.legend(loc='upper right')
    plt.show()
    for feat in sincere_stats.keys():
        get_plot(feat)