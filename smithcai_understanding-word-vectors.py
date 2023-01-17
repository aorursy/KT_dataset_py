!python -m spacy download en_vectors_web_lg

!python -m spacy link en_vectors_web_lg en_vectors_web_lg
from __future__ import unicode_literals

import spacy

nlp = spacy.load('en_vectors_web_lg')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import math

def distance2d(x1, y1, x2, y2):

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
distance2d(70, 30, 75, 40) # panda and capybara
distance2d(8, 3, 65, 90) # tarantula and elephant
import json
color_data = json.loads(open("/kaggle/input/xkcd.json").read())
def hex_to_int(s):

    s = s.lstrip("#")

    return int(s[:2], 16), int(s[2:4], 16), int(s[4:6], 16)
colors = dict()

for item in color_data['colors']:

    colors[item["color"]] = hex_to_int(item["hex"])
colors['olive']
colors['red']
colors['black']
import math

def distance(coord1, coord2):

    # note, this is VERY SLOW, don't use for actual code

    return math.sqrt(sum([(i - j)**2 for i, j in zip(coord1, coord2)]))

distance([10, 1], [5, 2])
def subtractv(coord1, coord2):

    return [c1 - c2 for c1, c2 in zip(coord1, coord2)]

subtractv([10, 1], [5, 2])
def addv(coord1, coord2):

    return [c1 + c2 for c1, c2 in zip(coord1, coord2)]

addv([10, 1], [5, 2])
def meanv(coords):

    # assumes every item in coords has same length as item 0

    sumv = [0] * len(coords[0])

    for item in coords:

        for i in range(len(item)):

            sumv[i] += item[i]

    mean = [0] * len(sumv)

    for i in range(len(sumv)):

        mean[i] = float(sumv[i]) / len(coords)

    return mean

meanv([[0, 1], [2, 2], [4, 3]])
distance(colors['red'], colors['green']) > distance(colors['red'], colors['pink'])
def closest(space, coord, n=10):

    closest = []

    for key in sorted(space.keys(),

                        key=lambda x: distance(coord, space[x]))[:n]:

        closest.append(key)

    return closest
closest(colors, colors['red'])
closest(colors, [150, 60, 150])
closest(colors, subtractv(colors['purple'], colors['red']))
closest(colors, addv(colors['blue'], colors['green']))
# the average of black and white: medium grey

closest(colors, meanv([colors['black'], colors['white']]))
# an analogy: pink is to red as X is to blue

pink_to_red = subtractv(colors['pink'], colors['red'])

closest(colors, addv(pink_to_red, colors['blue']))
# another example: 

navy_to_blue = subtractv(colors['navy'], colors['blue'])

closest(colors, addv(navy_to_blue, colors['green']))
import random

red = colors['red']

blue = colors['blue']

for i in range(14):

    rednames = closest(colors, red)

    bluenames = closest(colors, blue)

    print ("Roses are " + rednames[0] + ", violets are " + bluenames[0])

    red = colors[random.choice(rednames[1:])]

    blue = colors[random.choice(bluenames[1:])]
doc = nlp(open("/kaggle/input/pg345.txt").read())

# use word.lower_ to normalize case

drac_colors = [colors[word.lower_] for word in doc if word.lower_ in colors]

avg_color = meanv(drac_colors)

print (avg_color)
closest(colors, avg_color)
for cname in closest(colors, colors['mauve']):

    print (cname + " trousers")
# let's define documents with 1 or 2 sentences each.



documents =[

            "I was hungry ,so i ate a cake. Now I like cakes more.",

            "I ate chicken but I'm still hungry so I bought a tasty cake and gave it to my hungry sister",

            "My sister was hungry so she ate the tasty cake."

]
import re

import nltk

nltk.download('punkt')

nltk.download('stopwords')
# normalization

processed_documents = []

for document in documents:

    processed_text = document.lower()

    processed_text = re.sub('[^a-zA-Z]', ' ', processed_text )

    processed_documents.append(processed_text)

    

processed_documents
# tokenization

# note that we didn't use setence tokenization because we are considering the document wise

all_words = [nltk.word_tokenize(sent) for sent in processed_documents]

print(all_words)
# stopwords removal



from nltk.corpus import stopwords

for i in range(len(all_words)):

    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

all_words
from nltk.stem import WordNetLemmatizer 

  

lemmatizer = WordNetLemmatizer() 

final_docs = []

for words in all_words:

    doc = [lemmatizer.lemmatize(word) for word in words]

    final_docs.append(doc)

final_docs
# constructing vocab

vocab = []

for words in final_docs:

    for word in words:

        if not (word in vocab):

            vocab.append(word)
vocab
# Document Matrix

dm=[]

def initTemparray(length_vocab):

    a=[]

    for i in range(length_vocab):

        a.append(0)

      #print(a)

    return a



for doc in final_docs:

    temparray = initTemparray(len(vocab))

    for word in doc:

        #word = removeChars(word)

        if word in vocab:

            temparray[vocab.index(word)]= temparray[vocab.index(word)] + 1

    dm.append(temparray)

dm 

# dm means document matrix and dv means document vector

# Here, the rows correspond to the documents in the corpus and the columns correspond to the tokens in the dictionary. 
# Cosidering only the first 3 words in vocabulary for easy representation



vocab_wise = list(zip(*dm[::-1]))
vocab_wise
from matplotlib import pyplot

from mpl_toolkits.mplot3d import Axes3D

import random



# Since we can plot only 3 dimentions at a time we could take only the first 3 rows representing hungry, ate, cake.

x_vals, y_vals, z_vals = list(vocab_wise[0]), list(vocab_wise[1]), list(vocab_wise[2])

import plotly.express as px

fig = px.scatter_3d(x=x_vals, y=y_vals, z=z_vals, text=vocab[0:3])



fig.update_layout(scene = dict(

                    xaxis_title='hungry',

                    yaxis_title='ate',

                    zaxis_title='cake'),

                    width=1000,

                    margin=dict(r=20, b=5, l=5, t=5))



fig.show()
print(dm[0])

print(final_docs[0])

print(vocab)
print(dm[1])

print(final_docs[1])

print(vocab)
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
# Co-occurance matrix

# We build the map by taking into account each word, say hungry and finding the documents that has hungry (usage) and adding them 

cmatrix=[]

for word in vocab:

    current_index =vocab.index(word)

    temparray = initTemparray(len(vocab))

    print(f'Constructing for word: {word} = 1')

    for idx, dv in enumerate(dm):

        print('-'*45)

        print(f'{idx+1} row in dm')

        print(dv)

        if dv[current_index]==1:

            for i in range(len(vocab)):

                temparray[i]=temparray[i]+ dv[i]

                

    cmatrix.append(temparray)

    print(f'Final corpus map for {word}: {temparray}')

cmatrix
print(vocab)
final_docs
import copy

# normalized corpusmap

normCorpusMap = copy.deepcopy(cmatrix)

for j in range(len(vocab)):

    for i in range(len(vocab)):

        normCorpusMap[j][i] = cmatrix[j][i] - max(cmatrix[j])+1
normCorpusMap
from matplotlib import pyplot

from mpl_toolkits.mplot3d import Axes3D

import random



# Since we can plot only 3 dimentions at a time we could take only the first 3 rows representing hungry, ate, cake.

x_vals, y_vals, z_vals = normCorpusMap[0], normCorpusMap[1], normCorpusMap[2]

import plotly.express as px

fig = px.scatter_3d(x=x_vals, y=y_vals, z=z_vals, text=vocab)



fig.update_layout(scene = dict(

                    xaxis_title='hungry',

                    yaxis_title='ate',

                    zaxis_title='cake'),

                    width=1000,

                    margin=dict(r=20, b=10, l=10, t=10))



fig.show()
#sample test

tokens = nlp("dog cat banana afskfsd")



for token in tokens:

    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
doc = nlp(open("/kaggle/input/pg345.txt").read())

type(doc)
# all of the words in the text file

tokens = list(set([w.text for w in doc if w.is_alpha]))
len(tokens)
print(tokens[:10])
nlp.vocab
nlp.vocab['cheese'].vector
def vec(s):

    return nlp.vocab[s].vector
from numpy import dot

from numpy.linalg import norm



# cosine similarity

def cosine(v1, v2):

    if norm(v1) > 0 and norm(v2) > 0:

        return dot(v1, v2) / (norm(v1) * norm(v2))

    else:

        return 0.0
cosine(vec('dog'), vec('puppy')) > cosine(vec('trousers'), vec('octopus'))
def spacy_closest(token_list, vec_to_check, n=10):

    return sorted(token_list,

                  key=lambda x: cosine(vec_to_check, vec(x)),

                  reverse=True)[:n]
# what's the closest equivalent of basketball?

spacy_closest(tokens, vec("basketball"))
# halfway between day and night

spacy_closest(tokens, meanv([vec("day"), vec("night")]))
spacy_closest(tokens, vec("wine"))
spacy_closest(tokens, subtractv(vec("wine"), vec("alcohol")))
spacy_closest(tokens, vec("water"))
spacy_closest(tokens, addv(vec("water"), vec("frozen")))
# analogy: blue is to sky as X is to grass

blue_to_sky = subtractv(vec("blue"), vec("sky"))

spacy_closest(tokens, addv(blue_to_sky, vec("grass")))
def meanv(coords):

    # assumes every item in coords has same length as item 0

    sumv = [0] * len(coords[0])

    for item in coords:

        for i in range(len(item)):

            sumv[i] += item[i]

    mean = [0] * len(sumv)

    for i in range(len(sumv)):

        mean[i] = float(sumv[i]) / len(coords)

    return mean

meanv([[0, 1], [2, 2], [4, 3]])
def sentvec(s):

    sent = nlp(s)

    return meanv([nlp.vocab[word].vector for word in sent])
nlp.vocab["my"].vector
from __future__ import unicode_literals, print_function

from spacy.lang.en import English # updated



# creating a custom pipeline for getting sentenses alone



nlp = English()

nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated

doc = nlp(open("/kaggle/input/pg345.txt").read())

sentences = [sent.string.strip() for sent in doc.sents]
len(sentences)
def spacy_closest_sent(space, input_str, n=10):

    input_vec = sentvec(input_str)

    return sorted(space,

                  key=lambda x: cosine(np.mean([w.vector for w in x], axis=0), input_vec),

                  reverse=True)[:n]
#for sent in spacy_closest_sent(sentences, "My favorite food is strawberry ice cream."):

#    print (sent.text)

#    print ("---")