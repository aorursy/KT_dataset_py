import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

%matplotlib inline



train = pd.read_csv('../input/train_1.csv').fillna(0)

key = pd.read_csv('../input/key_1.csv')

key.info()

key

train.head()

train.info()

train





def get_language(page):

    res = re.search('[a-z][a-z].wikipedia.org',page)

    if res:

        return res[0][0:2]

    return 'na'



train['lang'] = train.Page.map(get_language)



from collections import Counter



print(Counter(train.lang))
lang_sets = {}

lang_sets['en'] = train[train.lang == 'en'].iloc[:,0:-1]

lang_sets['ja'] = train[train.lang == 'ja'].iloc[:,0:-1]

lang_sets['de'] = train[train.lang == 'de'].iloc[:,0:-1]

lang_sets['na'] = train[train.lang == 'na'].iloc[:,0:-1]

lang_sets['fr'] = train[train.lang == 'fr'].iloc[:,0:-1]

lang_sets['zh'] = train[train.lang == 'zh'].iloc[:,0:-1]

lang_sets['ru'] = train[train.lang == 'ru'].iloc[:,0:-1]

lang_sets['es'] = train[train.lang == 'es'].iloc[:,0:-1]







sums = {}

for key in lang_sets:

    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]

    

print(sums['en'])