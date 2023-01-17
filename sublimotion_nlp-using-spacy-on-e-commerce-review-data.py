# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets
import spacy
from spacy.util import minibatch, compounding
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read the file
dfRaw = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
#Basic introspection of the file
dfRaw.info()
dfRaw.head()
#Create a new dataframe with the field of interests and drops any null values.
dfRec = dfRaw[['Review Text', 'Recommended IND']].dropna()
dfRec.info()
dfRec.head()
#Format the dataset for use in spacy
dfRec['dataset'] = dfRec.apply(lambda row: (row['Review Text'],row['Recommended IND']), axis=1)
ecom = dfRec['dataset'].tolist()
ecom[5]
#helper functions
def load_data(limit=0, split=0.8):
    """Load data from the e-commerce dataset."""
    # Partition off part of the train data for evaluation
    train_data = ecom
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{'POSITIVE': bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}
#Initialize variable for the spacy model
model=None
n_iter=20 
n_texts=2000
#Create a new model
nlp = spacy.load('en')  # create blank Language class
print("Created blank 'en' model")
#Create text classifier, add to the pipeline and create label
textcat = nlp.create_pipe('textcat')
nlp.add_pipe(textcat, last=True)
# add label to text classifier
textcat.add_label('POSITIVE')
#Load the data and split the dataset into DEV and Train for evaluation
print("Loading ecom data...")
(train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
print("Using {} examples ({} training, {} evaluation)".format(n_texts, len(train_texts), len(dev_texts)))
train_data = list(zip(train_texts,[{'cats': cats} for cats in train_cats]))
#Run model
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))
output_dir = "./ecom_product_rec"
nlp.to_disk(output_dir)
# Load saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
test_pos = "I love, love, love this jumpsuit. it's fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments!"
doc2 = nlp2(test_pos)
print(test_pos, doc2.cats)
test_neg = "3 tags sewn in, 2 small (about 1'' long) and 1 huge (about 2'' x 3''). very itchy so i cut them out. then the thread left behind was plasticy and even more itchy! how can you make an intimates item with such itchy tags? not comfortable at all! also - i love bralettes and wear them all the time including to work. i am a b cup. however, this one is so thin and flimsy that it gives no support even to a b cup - so for me this would only be a lounging bralette - if it wasn't so itchy!"
doc3 = nlp2(test_neg)
print(test_neg, doc3.cats)