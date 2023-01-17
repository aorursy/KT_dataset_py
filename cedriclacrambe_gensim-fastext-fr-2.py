# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import gensim

import itertools

import random

import sys

from gensim.models import FastText

# Any results you write to the current directory are saved as output.
from smart_open import smart_open

from nltk import RegexpTokenizer

class token_stream(object):

    def __init__(self,fichier):

        self.fichier=fichier

    def __iter__(self):

        toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')

        with smart_open(self.fichier, encoding='utf8') as f:

            for l in f:

                yield toknizer.tokenize(l)

corpus_file='../input/textes.txt.bz2'

gen=iter(token_stream(corpus_file))

next(gen)
next(gen)
vocab_sizes=[1e6,200e3,60e3,2e6,1.5e6] 

dims= [64,100,256,300,512]

combinations=[(a,b) for a,b in itertools.product(dims,vocab_sizes)]

random.shuffle(combinations)

combinations


for dim,vocab in combinations:

    vocab=int(vocab)

    dim=int(dim)

    st=os.statvfs(".")

    if st.f_bavail*st.f_bsize<900e6:

        break

    array_size_float64=vocab*dim*8

    print(vocab,"x",dim,vocab*dim,f"{vocab//1e6:.0f}m")

    if array_size_float64<4e9:

        

        ft=FastText(#sentences=line_stream(corpus_file),

                    sg=1,

                    size=dim,

                    window=15,

                    min_count=5,

                    workers =os.cpu_count(),

                    negative=10,

                    #bucket=50000,

                    max_vocab_size=vocab,

                    sorted_vocab=1

                   )

        ft.build_vocab(sentences=token_stream(corpus_file))

        total_examples = ft.corpus_count

        ft.train(sentences=token_stream(corpus_file), total_examples=total_examples, epochs=10)

        

        if vocab>1e6:

            vocab_s=f"{vocab//1e6:.0f}m"

            if vocab%1e6//1000!=0:

                vocab_s=vocab_s+f"{vocab%1e6//1000:.0f}"

        elif vocab>1e3:

            vocab_s=f"{vocab//1e3:.0f}k"

        name=f"fastext_fr_{dim}_{vocab_s}"

        print(name)

        ft.save(name)
