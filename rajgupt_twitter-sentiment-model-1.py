# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train.shape
df = train.copy()

df = df.dropna(axis=0, subset=['text'])

df = df.fillna('')

df['select_len'] = df.selected_text.apply(lambda t: len(t.split()))

df['all_len'] = df.text.apply(lambda t: len(t.split()))

df['select_pct'] = df.apply(lambda r: r['select_len']/r['all_len']*100, axis=1)

print(df.head())

print(df.shape)


g = sns.FacetGrid(df, col="sentiment")

g.map(plt.hist, "select_pct");
g = sns.FacetGrid(df, col="sentiment")

g.map(plt.hist, "all_len");
g = sns.FacetGrid(df, col="sentiment")

g.map(plt.hist, "select_len");
positive_text_list = df[df.sentiment == 'positive']['selected_text'].tolist()

positive_text_list[:5]
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import nltk

import spacy

import random

from spacy.util import minibatch, compounding

from pathlib import Path
# Prepare Training Data for spacy nlp - ner 

def create_training_data(df: pd.DataFrame, sentiment: str):

    '''

    ref: https://spacy.io/usage/training#ner

    sample training data

    TRAIN_DATA = [

        ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),

        ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),

    ]

    '''

    train_data = []

    df_select = df[df.sentiment == sentiment]

    for i, row in df_select.iterrows():

        start = row['text'].find(row['selected_text'])

        train_sample = (row['text'], 

                        {"entities": [(start,

                                       start+len(row['selected_text']),

                                      "selected_text")]})

        train_data.append(train_sample)

    return train_data
LABEL = "selected_text"



def train(data, n_iter=10):

    """Set up the pipeline and entity recognizer, and train the new entity."""

    random.seed(0)

    nlp = spacy.blank("en")  # create blank Language class

    ner = nlp.create_pipe("ner")

    nlp.add_pipe(ner)

    ner.add_label(LABEL)  # add new entity label to entity recognizer

    optimizer = nlp.begin_training()

    move_names = list(ner.move_names)

    # get names of other pipes to disable them during training

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train NER

        sizes = compounding(1.0, 4.0, 1.001)

        # batch up the examples using spaCy's minibatch

        for itn in range(n_iter):

            random.shuffle(data)

            batches = minibatch(data, size=sizes)

            losses = {}

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)

            print("Losses", losses)

    return nlp
def save_model(nlp, sentiment, outdir='/kaggle/working/models'):

    # save model to output directory

    output_dir = Path('/kaggle/working/models/'+sentiment)

    if not output_dir.exists():

        output_dir.mkdir(parents=True, exist_ok=True)

    nlp.meta["name"] = sentiment  # rename model

    nlp.to_disk(output_dir)

    print("Saved model to", output_dir)
def is_model_trained(sentiment, outdir='/kaggle/working/models'):

    if os.path.exists(os.path.join(outdir, sentiment+'/meta.json')):

        return True

    return False
is_model_trained('nuetral')
for sent in df.sentiment.unique():

    train_data = create_training_data(df,sent)

    if not is_model_trained(sent):

        print(f'training {sent} tweets selected_text ner model...')

        model = train(train_data, n_iter=3)

        save_model(model, sent)
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test.head()
# load models

models = {}

models['positive'] = spacy.load('/kaggle/working/models/positive')

models['negative'] = spacy.load('/kaggle/working/models/negative')

models['neutral'] = spacy.load('/kaggle/working/models/neutral')
def predict_entities(text, model):

    doc = model(text)

    ent_array = []

    for ent in doc.ents:

        start = text.find(ent.text)

        end = start + len(ent.text)

        new_int = [start, end, ent.label_]

        if new_int not in ent_array:

            ent_array.append([start, end, ent.label_])

    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text

    return selected_text
predictions = []

for i, row in test.iterrows():

    selected_text = predict_entities(row['text'], models[row['sentiment']])

    predictions.append([row['textID'], selected_text])

df = pd.DataFrame(predictions)

df.columns=['textID', 'selected_text']

df.to_csv('submission.csv', index=False)