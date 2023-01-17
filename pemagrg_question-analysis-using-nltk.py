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
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()

stem = ps.stem



def getQueryVector(searchQuery):

    vector = {}

    for token in searchQuery:

        if token in stopwords:

            continue

        token = stem(token)

        if token in vector.keys():

            vector[token] += 1

        else:

            vector[token] = 1

    return vector



def getQueryVector(searchQuery):

    vector = {}

    for token in searchQuery:

        token = stem(token)

        if token in vector.keys():

            vector[token] += 1

        else:

            vector[token] = 1

    return vector
import nltk

from nltk.tree import Tree

from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()

stem = ps.stem

grammar = "NE:{<NN.*><.*><NN.*>| <NN.*><VB.*>}"

chunk_parser=nltk.RegexpParser(grammar)



#------------------QUESTION CLASSIFICATION--------



def getphrase(text):

    q1_tokens = nltk.word_tokenize(text)

    q1_pos = nltk.pos_tag(q1_tokens)

    grammar = "NE:{<NN.*><.*><VBG>|<VB.*><.*><RP.*>|<NN.*><.*><NN.*>}"

    chunk_parser = nltk.RegexpParser(grammar)

    chunk_tree = chunk_parser.parse(q1_pos)



    phrase_list = []

    for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'NE'):

        w = (subtree.leaves())

        for a, b in w:

            phrase_list.append(a)

    phrase = " ".join(phrase_list)

    return (phrase)





def getContinuousChunk(text):

    chunks = []

    answerToken = nltk.word_tokenize(text)

    nc = nltk.pos_tag(answerToken)



    prevPos = nc[0][1]

    entity = {"pos": prevPos, "chunk": []}

    for c_node in nc:

        (token, pos) = c_node

        if pos == prevPos:

            prevPos = pos

            entity["chunk"].append(token)

        elif prevPos in ["DT", "JJ"]:

            prevPos = pos

            entity["pos"] = pos

            entity["chunk"].append(token)

        else:

            if not len(entity["chunk"]) == 0:

                chunks.append((entity["pos"], " ".join(entity["chunk"])))

                entity = {"pos": pos, "chunk": [token]}

                prevPos = pos

    if not len(entity["chunk"]) == 0:

        chunks.append((entity["pos"], " ".join(entity["chunk"])))

    return chunks



def getNamedEntity(answers):

    chunks = []

    for answer in answers:

        answerToken = nltk.word_tokenize(answer)

        nc = nltk.ne_chunk(nltk.pos_tag(answerToken))

        entity = {"label":None,"chunk":[]}

        for c_node in nc:

            if(type(c_node) == Tree):

                if(entity["label"] == None):

                    entity["label"] = c_node.label()

                entity["chunk"].extend([ token for (token,pos) in c_node.leaves()])

            else:

                (token,pos) = c_node

                if pos == "NNP":

                    entity["chunk"].append(token)

                else:

                    if not len(entity["chunk"]) == 0:

                        chunks.append((entity["label"]," ".join(entity["chunk"])))

                        entity = {"label":None,"chunk":[]}

        if not len(entity["chunk"]) == 0:

            chunks.append((entity["label"]," ".join(entity["chunk"])))

    if len(chunks) == 0:

        answerToken = nltk.word_tokenize(answers)

        nc = nltk.ne_chunk(nltk.pos_tag(answerToken))

        for word,pos in nc:

            if pos == "NNP" or pos == "NN" or pos == "VB":

                chunks.append(word)

    return chunks
import nltk



def determineAnswerType(text):

    word = nltk.word_tokenize(text)

    pos_tag = nltk.pos_tag(word)

    chunk = nltk.ne_chunk(pos_tag)

    ne_list = []

    for ele in chunk:

        if isinstance(ele, nltk.Tree):

            ne_list.append(ele.label())

    if len(ne_list)>1:

        return ne_list

    else:

        questionTaggers = ['WP', 'WDT', 'WP$', 'WRB', 'VBZ']



        qPOS = nltk.pos_tag(nltk.word_tokenize(text))

        qTag = None



        for token in qPOS:

            if token[1] in questionTaggers:

                qTag = token[0].lower()

                break



        if qTag == None:

            if len(qPOS) > 1:

                if qPOS[0][0].lower() in ['is', 'are', 'can', 'should','will']:

                    qTag = "YESNO"

                    return "YESNO"



        if qTag != "YESNO":

            # who/where/what/why/when/is/are/can/should

            if qTag == "who":

                return "PERSON"

            elif qTag == "where":

                return "LOCATION"

            elif qTag == "when":

                return "DATE"

            elif qTag == "which":

                if len(qPOS) > 1:

                    t2 = qPOS[1]

                    if t2[0].lower() in ["year", "day", "date", "week", "month"]:

                        return "DATE"

                    elif t2[0].lower() in ["city", "state", "country"]:

                        return "LOCATION"

                    elif t2[0].lower() in ["person", "man", "women", "uncle", "aunt", "male", "female"]:

                        return "PERSON"

            elif qTag == "what":

                qTok = getContinuousChunk(text)

                '''if len(qTok) > 1:

                    if qTok[1][1] in ['is', 'are', 'was', 'were'] and qTok[2][0] in ["NN", "NNS", "NNP", "NNPS"]:

                        text = " ".join([qTok[0][1], qTok[2][1], qTok[1][1]])

                        print("DEFINITION")'''

                for token in qPOS:

                    if token[0].lower() in ["city", "place", "country", "capital", "state", "location", "area","route"]:

                        return "LOCATION"

                    elif token[0].lower() in ["company", "industry", "organization"]:

                        return "ORGANIZATION"

                    elif token[0].lower() in ["cost", "area", "number"]:

                        return "NUMBER"

            elif qTag == "how":

                t2 = qPOS[1]

                if t2[0].lower() in ["few", "great", "little", "many", "much"]:

                    return "QUANTITY"

                elif t2[0].lower() in ["tall", "wide", "big", "far"]:

                    return "LINEAR_MEASURE"

                return "1FULL"

            else:

                return "FULL"

import nltk

text = "where is Pema Gurung"

word_tokens = nltk.word_tokenize(text)

pos_tag = nltk.pos_tag(word_tokens)

chunk = nltk.ne_chunk(pos_tag)

ne_list = []

for ele in chunk:

    if isinstance(ele,nltk.Tree):

        ne_list.append(ele.label())

if len(ne_list)>1:

    print(ne_list)

print(ne_list)
import nltk





##-----Main----

isActive = True

while isActive:

    text = input('Enter your query: ')

    quest = determineAnswerType(text)

    print ("---question classified: ",quest)