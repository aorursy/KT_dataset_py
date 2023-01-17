# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

import networkx as nx

from itertools import combinations

from collections import defaultdict

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/quora-insincere-questions-classification/train.csv", nrows=1000, encoding="ISO-8859-1")

#data = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

data.head()

#common entities list would be like the one that we are getting it 

#edge of co-ccurance



def cooccurrence(common_entities):

    com = defaultdict(int)

    #Build co-occurence matrix

    for w1,w2 in combinations(sorted(common_entities),2):

        com[w1,w2] += 1

    result = defaultdict(dict)

    for (w1,w2), count in com.items():

        if w1 != w2:

            #result[w1][w2]={'weight':count}

            result[w1][w2]= count

    return result



def combine_nested_dicts(x, y):

    all_keys = set().union(x.keys(), y.keys())

    print(all_keys)

    for key in all_keys:

        if key in y and key in x :

            x[key] = Counter(x[key]) + Counter(y[key])

            x[key] = dict(x[key])

            print(x[key])

        else:

            x.update(y[key])

    for key in y:

        if key not in x:

            x.update(y[key])

    return (x)
# test coocurrence 

print(cooccurrence('abcaqwvv'))



# test combining dicts

x = cooccurrence('abb')

y = cooccurrence('acbd')

z = cooccurrence('efg')

print(combine_nested_dicts(x, z))
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

for ent in doc.ents:

    print(ent.text, ent.start_char, ent.end_char, ent.label_)

nlp = spacy.load('en_core_web_sm') #dictionary of words that contains english words conent from 

docs = nlp.pipe(data)

#cooccurrence(docs.ents)

#initialize claim counter and 

claim_counter = 0

list_indexes = []

list_entities = []

for doc in docs:

    #print(doc)

    for ent in doc.ents:

        print (ent)

        list_indexes.append(claim_counter)

        list_entities.append(ent.text)

        print(ent.text)

        print(ent.start_char)

        print(ent.end_char)

        print(ent.label_)

    claim_counter +=1
import h2o

h2o.init()
train_data=h2o.import_file("/input/glove480small-rao/train_glove480_small.csv")
import pandas as pd

train_paragram_small = pd.read_csv("../input/train_paragram_small.csv")
import pandas as pd

train_google_news_small = pd.read_csv("../input/train_google_news_small.csv")