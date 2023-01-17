import numpy as np

import pandas as pd

from IPython.display import  display

from collections import defaultdict

from itertools import combinations

pd.set_option('display.max_colwidth',-1)

import textblob as tb

import nltk as nl

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn import ensemble

from sklearn import metrics
# let's define union-find function

def indices_dict(lis):

    d = defaultdict(list)

    for i,(a,b) in enumerate(lis):

        d[a].append(i)

        d[b].append(i)

    return d



def disjoint_indices(lis):

    d = indices_dict(lis)

    sets = []

    while len(d):

        que = set(d.popitem()[1])

        ind = set()

        while len(que):

            ind |= que 

            que = set([y for i in que 

                         for x in lis[i] 

                         for y in d.pop(x, [])]) - ind

        sets += [ind]

    return sets



def disjoint_sets(lis):

    return [set([x for i in s for x in lis[i]]) for s in disjoint_indices(lis)]

train_df=pd.read_csv('../input/train.csv')

# only duplicated questions

ddf=train_df[train_df.is_duplicate==1]

print('Duplicates questions shape:',ddf.shape)

# get all duplicated questions

clean_ddf1=ddf[['qid1','question1']].drop_duplicates()

clean_ddf1.columns=['qid','question']

clean_ddf2=ddf[['qid2','question2']].drop_duplicates()

clean_ddf2.columns=['qid','question']

all_dqdf=clean_ddf1.append(clean_ddf2,ignore_index=True)

print(all_dqdf.shape)

# groupby qid1, and then we get all the combinations of id in each group

dqids12=ddf[['qid1','qid2']]

df12list=dqids12.groupby('qid1', as_index=False)['qid2'].agg({'dlist':(lambda x: list(x))})

print(len(df12list))

d12list=df12list.values

d12list=[[i]+j for i,j in d12list]

# get all the combinations of id, like (id1,id2)...

d12ids=set()

for ids in d12list:

    ids_len=len(ids)

    for i in range(ids_len):

        for j in range(i+1,ids_len):

            d12ids.add((ids[i],ids[j]))

print(len(d12ids))

# the same operation of qid2

dqids21=ddf[['qid2','qid1']]

display(dqids21.head(2))

df21list=dqids21.groupby('qid2', as_index=False)['qid1'].agg({'dlist':(lambda x: list(x))})

print(len(df21list))

ids2=df21list.qid2.values

d21list=df21list.values

d21list=[[i]+j for i,j in d21list]

d21ids=set()

for ids in d21list:

    ids_len=len(ids)

    for i in range(ids_len):

        for j in range(i+1,ids_len):

            d21ids.add((ids[i],ids[j]))

len(d21ids)

# merge two set

dids=list(d12ids | d21ids)

len(dids)

# split data into groups, so that each question in each group are duplicated

did_u=disjoint_sets(dids)

new_dids=[]

for u in did_u:

    new_dids.extend(list(combinations(u,2)))

len(new_dids)

new_ddf=pd.DataFrame(new_dids,columns=['qid1','qid2'])

print('New duplicated shape:',new_ddf.shape)

display(new_ddf.head(2))

# merge with all_dqdf to get question1 description

new_ddf=new_ddf.merge(all_dqdf,left_on='qid1',right_on='qid',how='left')

new_ddf.drop('qid',inplace=True,axis=1)

new_ddf.columns=['qid1','qid2','question1']

new_ddf.drop_duplicates(inplace=True)

print(new_ddf.shape)

new_ddf.head(2)

# the same operation with qid2

new_ddf=new_ddf.merge(all_dqdf,left_on='qid2',right_on='qid',how='left')

new_ddf.drop('qid',inplace=True,axis=1)

new_ddf.columns=['qid1','qid2','question1','question2']

new_ddf.drop_duplicates(inplace=True)

print(new_ddf.shape)

new_ddf.head(2)

# is_duplicate flag

new_ddf['is_duplicate']=1

new_ddf.head(2)

# let random select 10 rows to check the result

new_ddf.sample(10)

# the orininal duplicated pairs count:

print(len(all_dqdf))

# after we generate more data, then the duplicated pairs count:

print(len(new_ddf))

trainDF = new_ddf.append(train_df[train_df.is_duplicate==0].drop('id', 1))
#len(trainDF['question2'])

len(trainDF['question1'].dropna())
#building a vocabulary

vect = CountVectorizer()

questions = trainDF['question1'].append(trainDF['question2'])

q = vect.fit_transform(questions.values.astype('U'))
#bulding training data

q1 = vect.transform(trainDF['question1'].values.astype('U'))

q2 = vect.transform(trainDF['question2'].values.astype('U'))

#Training a decision tree on distance of q1-q2 matrices

X_train = q1-q2

Y_train = trainDF['is_duplicate'].values

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

dtree = tree.DecisionTreeClassifier(max_depth = 10)

dtree = dtree.fit(x_train, y_train)

dtree.score(x_test,y_test)
from nltk import word_tokenize          

from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):

        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



vect = CountVectorizer(ngram_range = (1,2), token_pattern=r'\b\w+\b', \

                       analyzer = 'word', encoding='ascii',strip_accents = 'ascii',\

                       tokenizer=LemmaTokenizer()) 



c = vect.fit_transform(['Bi-grams are cool?', 'more hi.', 'kake metros'])

c.toarray()


len(['bi-grams',

 'are',

 'cool',

 '?',

 'more',

 'hi',

 'kake',

 'metro',

 'bi-grams are',

 'are cool',

 'cool ?',

 '? more',

 'more hi',

 'hi kake',

 'kake metro'])