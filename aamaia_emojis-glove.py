import numpy as np

import pandas as pd

import json

import spacy

from IPython.core.display import display, HTML



def load_emojis():

    rows = []

    with open('../input/emojinet/emojis.json') as f:

        for emoji in json.loads(f.read()):

            rows.append([emoji['name'], emoji['unicode'], ' '.join(emoji['keywords'])])    

    return np.array(rows)

    

emojis = load_emojis()

pd.DataFrame(emojis, columns=['name', 'unicode', 'keywords']).head()


nlp = spacy.load('en')

from tqdm import tqdm



with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt', 'r') as f:

    for line in tqdm(f, total=400000):

        parts = line.split()

        word = parts[0]

        vec = np.array([float(v) for v in parts[1:]], dtype='f')

        nlp.vocab.set_vector(word, vec)
docs = [nlp(str(keywords)) for _, _, keywords in tqdm(emojis)]

doc_vectors = np.array([doc.vector for doc in docs])
from numpy import dot

from numpy.linalg import norm



def most_similar(vectors, vec):

    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))

    dst = np.dot(vectors, vec) / (norm(vectors) * norm(vec))

    return np.argsort(-dst)

def query(v, most_n=5):

    ids = most_similar(doc_vectors, v)[:most_n]

    print(ids)

    html = []

    for name, unicode, keywords in emojis[ids]:

        values = unicode.split(' ')

        for v in values:

            c = chr(int(v.replace('U+', ''), 16))

            print(c, name)

            html.append(c)

    display(HTML('<font size="+3">{}</font>'.format(' '.join([x for x in html]))))

    


v = nlp(u'star'.lower()).vector

query(v)


v = nlp(u'star').vector + nlp(u'movie').vector

query(v)


v1 = nlp(u'animal').vector

query(v1, 10)
v1 = nlp(u'heart').vector - nlp(u'poker').vector

query(v1, 10)
v1 = nlp(u'mouse').vector + nlp(u'computer').vector

query(v1, 10)
v1 = nlp(u'mouse').vector - nlp(u'computer').vector

query(v1, 10)