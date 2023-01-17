# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
list1 =["European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices","The sick childs from Mumbai in the corner need Shyam","I eat a pizza at Pizzahut and Dominos with olives in Chennai","The brown fox is quick and he is jumping over the lazy dog named Rambo","A black television and a white stove were bought for the new apartment of Ram and Shyam","John hit the ball very hard","The little dog Cookie barked at the cat named Kitty","The green dog Rambo ate a large cookie on the table","The cells of Freddie Mercury were previously infected with HIV-AIDS","He is not allowed to appear in the exam of N.Y.C","Economic news had little effect on B.S.E and financial market"]

print(len(list1))
import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk import Tree, pos_tag, ne_chunk

from nltk.sem.relextract import NE_CLASSES

def preprocess(sent):

    sent = nltk.word_tokenize(sent)

    sent = nltk.pos_tag(sent)

    return sent
def chunking(sent):

    pattern = 'NP: {<DT>?<JJ>*<NN>}'

    cp = nltk.RegexpParser(pattern)

    cs = cp.parse(sent)

    return(sent)

    
def parsing_nltk(sent):

    words = nltk.word_tokenize(sent)

    tagged = nltk.pos_tag(words)

    chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

    chunkParser = nltk.RegexpParser(chunkGram)

    chunked = chunkParser.parse(tagged)

    print(chunked)
def ner(sent):

    ne_tree = ne_chunk(pos_tag(word_tokenize(sent)))

    print(ne_tree)
def namedentity(sent):

    tagged_sent = ne_chunk(pos_tag(sent.split()))

    print(tagged_sent)

    ace_tags = NE_CLASSES['ace']

    for node in tagged_sent:

         if type(node) == Tree and node.label() in ace_tags:

                words, tags = zip(*node.leaves())

                print (node.label() + '\t' +  ' '.join(words))

    #retur(tagged_sent)

    
j=1

for i in list1:

    print(str(j)+". "+i)

    print("POS TAG OF USING NLTK ")

    print(preprocess(i))

    print("\n")

    print("SHALLOW PARSING OF SENTENCE USING NLTK")

    print(parsing_nltk(i))

    print("\n")

    print("NAMED ENTITY RECOGNITION OF SENTENCE USING NLTK")

    ner(i)

    print("-----------------------------------------------")

    j+=1
import spacy

from spacy import displacy

nlp = spacy.load('en_core_web_sm')


def spacyparser(i):

    print('Original Sentence: %s' % (i))

    doc = nlp(i)

    print('POS TAGGING USING SPACY')

    for token in doc:

        print(token.text, token.tag_)

    print("\n")

    print('DEPENDECY PARSING USING SPACY')

    for chunk in doc.noun_chunks:

        print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)

    displacy.render(doc, style='dep', jupyter=True, options={'distance': 50})

    print("\n")

    print('\nNAMED ENTITY RECOGNITION USING SPACY')

    for ent in doc.ents:

        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    #displacy.render(doc, style='ent', jupyter=True)
spacyparser(list1[1])

    
spacyparser(list1[0])

spacyparser(list1[2])
spacyparser(list1[2])
spacyparser(list1[3])
spacyparser(list1[4])
spacyparser(list1[5])
spacyparser(list1[6])
spacyparser(list1[7])
spacyparser(list1[8])
spacyparser(list1[9])
spacyparser(list1[10])