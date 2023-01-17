import pandas as pd

import matplotlib.pyplot as plt

import spacy

import re

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

sample = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

def text_cleaner(text):

    """

    Cleans the text with some usual patterns.

    """

    text = text.lower()

    text = re.sub(r"https?:\/\/.*?( |$)", r"", text)

    text = re.sub(r"<.*?>", r"", text)

    text = re.sub(r"[/(){}\[\]\|@,;]", r"", text)

    #text = re.sub(r"[^0-9a-z #+_]", r"", text)

    text = text.strip()

    return text





train["text"] = train["text"].apply(lambda x: text_cleaner(str(x)))

test["text"] = test["text"].apply(lambda x: text_cleaner(str(x)))

train
train
train_pos_mask = [train['sentiment'] == 'positive']

train_positive = train[train_pos_mask[0]]

train_neg_mask = [train['sentiment'] == 'negative']

train_negative = train[train_neg_mask[0]]

train_neutral_mask = [train['sentiment'] == 'neutral']

train_neutral = train[train_neutral_mask[0]]
test_pos_mask = [test['sentiment'] == 'positive']

test_positive = test[test_pos_mask[0]]

test_neg_mask = [test['sentiment'] == 'negative']

test_negative = test[test_neg_mask[0]]

test_neutral_mask = [test['sentiment'] == 'neutral']

test_neutral = test[test_neutral_mask[0]]

def initial_look(df):



    print("Number of samples:", df.shape[0])

    print("")

    columns = ["text", "selected_text"]

    for column in columns:

        feature = df[column]

        print(column)

        print("number of NAs:", feature.isna().sum())

        print("Unique values:", feature.nunique())

        if column == "selected_text":

            print("Most common selected_text:")

            print(feature.value_counts(ascending=False)[:100])

        print("")





    
initial_look(train_positive)


train_positive["selected_text"].value_counts()[:30].plot.bar(figsize=(15,10), title="Count of selected_text")
train_positive["selected_text"].apply(lambda x: len(str(x))).value_counts()[:20].plot.bar(figsize=(15,10), title="Length of the selected text")
initial_look(train_negative)
plt.figure(figsize=(20,10))

train_negative["selected_text"].value_counts()[:30].plot.bar(figsize=(15,10), title="Count of selected_text")
train_negative["selected_text"].apply(lambda x: len(str(x))).value_counts()[:20].plot.bar(figsize=(15,10), title="Length of the selected text")
initial_look(train_neutral)
train_neutral["selected_text"].apply(lambda x: len(str(x))).value_counts()[:20].plot.bar(figsize=(15,10), title="Length of the selected text")
def create_training_data_for_spacy(df):

    """

    Example

    TRAIN_DATA = [

    (" haha better drunken tweeting you mean? ", {"entities": [(6, 12, "SELECTED_TEXT")]}),

    ("had an awsome salad! I recommend getting the Spicey buffalo chicken salad!", {"entities": [(0, 20, "SELECTED_TEXT")]}),

    """

    train_data =[]

    for i in range(df.shape[0]):

        selected_text_start = str(df.iloc[i, 1]).find(str(df.iloc[i, 2]))

        selected_text_end = selected_text_start + len(str(df.iloc[i, 2]))

        train_data.append((df.iloc[i, 1], {"entities":[(selected_text_start, selected_text_end, "SELECTED_TEXT")]}))



    return train_data
train_spacy_positive = create_training_data_for_spacy(train_positive)

train_spacy_negative = create_training_data_for_spacy(train_negative)
train_spacy_positive
#!/usr/bin/env python

# coding: utf8

"""Example of training spaCy's named entity recognizer, starting off with an

existing model or a blank model.



For more details, see the documentation:

* Training: https://spacy.io/usage/training

* NER: https://spacy.io/usage/linguistic-features#named-entities



Compatible with: spaCy v2.0.0+

Last tested with: v2.1.0

"""

from __future__ import unicode_literals, print_function



import plac

import random

from pathlib import Path

import spacy

from spacy.util import minibatch, compounding

import os



def spacy_ner_train(train_data, model=None, output_dir=None, n_iter=None):

    """Load the model, set up the pipeline and train the entity recognizer."""

    if model is not None:

        nlp = spacy.load(model)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")



    # create the built-in pipeline components and add them to the pipeline

    # nlp.create_pipe works for built-ins that are registered with spaCy

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe("ner")



    # add labels

    for _, annotations in train_data:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])



    # get names of other pipes to disable them during training

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train NER

        # reset and initialize the weights randomly – but only if we're

        # training a new model

        if model is None:

            nlp.begin_training()

        for itn in range(n_iter):

            random.shuffle(train_data)

            losses = {}

            # batch up the examples using spaCy's minibatch

            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(

                    texts,  # batch of texts

                    annotations,  # batch of annotations

                    drop=0.5,  # dropout - make it harder to memorise data

                    losses=losses,

                )

            print("Losses", losses)





    # save model to output directory

    if output_dir is not None:

        #output_dir = Path(output_dir)

        if not Path(output_dir).exists():

            os.makedirs(output_dir)

        

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)







spacy_ner_train(train_spacy_positive, model=None, output_dir="../models/model_positive", n_iter=5)
train_spacy_negative
spacy_ner_train(train_spacy_negative, model=None, output_dir="../models/model_negative", n_iter=5)
model_positive = spacy.load("../models/model_positive")

model_negative = spacy.load("../models/model_negative")


        

def predict_entities(text, model):

    doc = model(text)

    ent_list = []

    for ent in doc.ents:

        start = text.find(ent.text)

        end = start + len(ent.text)

        new_int = [start, end, ent.label_]

        if new_int not in ent_list:

            ent_list.append([start, end, ent.label_])

    selected_text = text[ent_list[0][0]: ent_list[0][1]] if len(ent_list) > 0 else text

    return selected_text         

            
selected_text_positive_predicted = test_positive["text"].apply(lambda x: predict_entities(x, model_positive))

selected_text_negative_predicted = test_negative["text"].apply(lambda x: predict_entities(x, model_negative))

selected_text_neutral_predicted = test_neutral["text"]
test["selected_text"] = selected_text_positive_predicted.append(selected_text_negative_predicted).append(selected_text_neutral_predicted)


sample["selected_text"] = test["selected_text"]

sample
sample.to_csv("submission.csv", index=False)