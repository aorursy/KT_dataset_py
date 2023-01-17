import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

import re

from nltk.corpus import stopwords



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)
train = pd.read_csv('../input/train.csv')
train.head()
print("The dataset contains", len(train), "items.")
train.index = train['id']

x_train = train['comment_text']

y_train = train.iloc[:, 2:]
y_train['clean'] = 1 - y_train.sum(axis=1) >= 1  

# beginner note: if some kind of toxicity is detected, the sum across rows will yield one, 

# and the subtraction will give zero, and one otherwise
kinds, counts = zip(*y_train.sum(axis=0).items())

# another beginner note: the sum operation yield a series, and a series behaves like a dictionary

# as it has the items function that returns index-value tuples.
bars = go.Bar(

        y=counts,

        x=kinds,

    )



layout = go.Layout(

    title="Class distribution in train set"

)



fig = go.Figure(data=[bars], layout=layout)

iplot(fig, filename='bar')
for kind in y_train.columns:

    print('Sample from "{}"'.format(kind))

    x_kind = x_train[y_train[kind]==1].sample(3)

    print("\n".join(x_kind))

    print("\n")
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

stops = stopwords.words("english")
def normalize(comment, lowercase, remove_stopwords):

    if lowercase:

        comment = comment.lower()

    comment = nlp(comment)

    lemmatized = list()

    for word in comment:

        lemma = word.lemma_.strip()

        if lemma:

            if not remove_stopwords or (remove_stopwords and lemma not in stops):

                lemmatized.append(lemma)

    return " ".join(lemmatized)
x_train_lemmatized = x_train.apply(normalize, lowercase=True, remove_stopwords=True)
x_train_lemmatized.sample(1).iloc[0]
from collections import Counter

word_counts = dict()



for kind in y_train.columns:

    word_counts[kind] = Counter()

    comments = x_train_lemmatized[y_train[kind]==1]

    for _, comment in comments.iteritems():

        word_counts[kind].update(comment.split(" "))
def most_common_words(kind, num_words=15):

    words, counts = zip(*word_counts[kind].most_common(num_words)[::-1])

    bars = go.Bar(

        y=words,

        x=counts,

        orientation="h"

    )



    layout = go.Layout(

        title="Most common words of the class \"{}\"".format(kind),

        yaxis=dict(

            ticklen=8  # to add some space between yaxis labels and the plot

        )

    )



    fig = go.Figure(data=[bars], layout=layout)

    iplot(fig, filename='bar')
most_common_words("toxic")
most_common_words("severe_toxic")
most_common_words("threat")
most_common_words("clean")