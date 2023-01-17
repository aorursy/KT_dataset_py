# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import nltk
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import nltk
import re
import pprint
from nltk import Tree


grammar = """
    NP: {<JJ>*<NN*>+}
    {<JJ>*<NN*><CC>*<NN*>+}
    {<DT>*<JJ>*<NN>+}
    {<DT>*<JJ>*<NNS>+}
    {<JJ>*<NNS>+}
    """


NPchunked = nltk.RegexpParser(grammar)

def chunked_text(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    sentences = [NPchunked.parse(sent) for sent in sentences]
    return sentences



def NP_chunked(sentences):
    nps = []
    NPchunked = nltk.RegexpParser(grammar)
    for sent in sentences:
        tree = NPchunked.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                t = subtree
                t = ' '.join(word for word, tag in t.leaves())
                nps.append(t)
    return nps



para = "Today is a very great day. Indian politicians are very corrupt."
sentences = chunked_text(para)
np =NP_chunked(sentences)
print(np)