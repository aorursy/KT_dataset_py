import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#test = pd.read_csv("../input/test.csv")
import re



def remove_tags(s):

    return re.sub('<[^>]*>', ' ', s)
from nltk import pos_tag

from nltk.stem.lancaster import LancasterStemmer



word_re = re.compile('([a-zA-Z]+(-[a-zA-Z]+)?(\'[a-zA-Z]+)?)')



st = LancasterStemmer()



def get_words(s):

    s = remove_tags(s)

    return pos_tag(list(map(lambda x: x[0].lower(), word_re.findall(s))))

        



def calc_word_count(s, d):

    for word in get_words(s):

        if word in d.keys():

            d[word] += 1

        else:

            d[word] = 1



def make_dataseries(d):

    lst = sorted(d.items(), key=lambda x: x[1], reverse=True)

    tagged_words, cnts  = zip(*lst)

    words, tags = zip(*tagged_words)

    df = pd.DataFrame(data={

        'word': words,

        'cnt': cnts,

        'tag': tags,

    }, index=range(len(words)))

    return df

            

def get_words_from_dataset(dataset):

    res = dict()

    for index, row in list(dataset.iterrows())[:100]:

        calc_word_count(row['title'], res)

        #calc_word_count(row['content'], res)

    return make_dataseries(res)
diy = pd.read_csv("../input/diy.csv")



tags = set(["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS"])



r = get_words_from_dataset(diy)



r.loc[r['tag'].isin(tags)]
