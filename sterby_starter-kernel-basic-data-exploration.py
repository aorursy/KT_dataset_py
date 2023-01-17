import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
df = pd.read_json("../input/recipes.json")
df.head()
print("We have {} recipes".format(df.shape[0]))
!python -m spacy download de_core_news_sm
import spacy

nlp = spacy.load('de_core_news_sm', disable=['parser', 'tagger', 'ner'])
tokenized = [nlp(t) for t in df['Instructions'].values]
for t in tokenized[0]:

    print(t)
vocab = {}

for txt in tokenized:

    for token in txt:

        if token.text not in vocab.keys():

            vocab[token.text] = len(vocab)
print("Number of unique tokens: {}".format(len(vocab)))