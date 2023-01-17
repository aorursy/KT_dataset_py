#!/usr/bin/env python

# coding: utf-8

# Writer - AMAnOSwal

# In[ ]:



#Defining the English Phonemes as in CMU Dict



phonemeE ='AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'

phonemeE = [o.strip() for o in phonemeE.split(' ')]



#Above English Phonemes convrted to Hindi Phonemes



phonemeH = 'आ ऐ यह औ ाव आय बी च डी ढ यह ेर ेय फ ग हह यह ीय झ क ल म न ंग ऊ ोय प र स श टी तह यह उव व् व य ज़ ज़ह'



#Mapping Created



import numpy as np

import pandas as pd

mapping = {}

for (i,j) in zip(phonemeE,phonemeH.split(' ')):

    mapping[i] = j



# Function defined to transliterate a single word using mappings created above



def convert(english):

    h = []

    for i in a:

        h.append(x[i])

    return ''.join(h)



#Function defined to break a word into its corresponding phonemes



import nltk

from functools import lru_cache

from itertools import product as iterprod



try:

    arpabet = nltk.corpus.cmudict.dict()

except LookupError:

    nltk.download('cmudict')

    arpabet = nltk.corpus.cmudict.dict()



@lru_cache()

def wordbreak(s):

    s = s.lower()

    if s in arpabet:

        return arpabet[s]

    middle = len(s)/2

    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)

    for i in partition:

        pre, suf = (s[:i], s[i:])

        if pre in arpabet and wordbreak(suf) is not None:

            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]

    return None

wordbreak('mississipi')



#Text loaded and preprocessed for further use



a = open("../input/wordsnlp/en-hi.txt")

x = []

for i in a:

    o = i.split('\t')

    x.append([o[0], o[1][:-1]])





#Utility function



def use(a):

    import string

    f = []

    for i in a:

        f.append(i.rstrip(string.digits))

    return f



#Using loop to convert words individually using wordbreak(), use() and created mappings



enga = []

nc = []

for i in range(len(x)):

    try:

        o = x[i][0]

        w = wordbreak(o)[0]

        usefulP = use(w)

        enga.append(usefulP)

    except:

        nc.append(i)



a = []

for i in enga:

    b = []

    for j in i:

        b.append(mapping[j])

    a.append(b)



candidate = []

for i in a:

    candidate.append(''.join(i))



# Creating Reference for Performance Calculations 



reference = []

for i in range(len(x)):

    if(i not in nc):

        reference.append(x[i][1])



# Calculating BLEU Score



from nltk.translate.bleu_score import corpus_bleu

score = 0

for i in range(len(reference)):

    score+=corpus_bleu([reference[i]], [candidate[i]], weights=(1, 0, 0, 0))

print("Average Bleu Score", score/len(reference))


