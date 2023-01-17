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
#header = infer by default
rawtraindf = pd.read_csv('../input/train.tsv', delimiter="\t")
rawtestdf = pd.read_csv('../input/test.tsv', delimiter="\t")
print(rawtraindf.shape) # 156060,4
print(rawtraindf.head(2))
colnames = rawtraindf.columns
print(colnames) # PhraseId, SentenceId, Phrase, Sentiment
#for index, row in rawtraindf.iterrows():
#    print(row)
rawtraindf.describe()
#sentiment varies from 0 to 4
#https://ep2018.europython.eu/conference/talks/introduction-to-sentiment-analysis-with-spacy
import spacy
nlp_en = spacy.load('en')
#general experiments ignore this section
#for index, row in rawtraindf.iterrows():
#    phrase = row[2]
#    print(type(phrase))
#    document = nlp_en(phrase)
#    for sentence in document.sents:
#        print(sentence)
        
#    print(document.sentiment)
#    print("Entries")
#    for ent in document.ents:
#        print(ent, ent.label, ent.label_)
#    if (index == 0):
#        break
#https://github.com/explosion/spacy/blob/master/examples/training/train_textcat.py

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(rawtraindf,test_size=0.2)
print(train_df.shape) # 124k
print(val_df.shape) # 312
train_texts = train_df['Phrase']
train_cats = train_df['Sentiment']
val_texts = val_df['Phrase']
val_cats = val_df['Sentiment']
#print(type(train_cats)) # series
train_cats = train_cats.reset_index(drop=True)
#with pd.option_context('display.max_rows', None,'display.max_columns', None):
#    print(train_cats)

train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
val_cats = val_cats.reset_index(drop=True)
#The sentiment labels are:
#0 - negative
#1 - somewhat negative
#2 - neutral
#3 - somewhat positive
#4 - positive

def getSentimentStr(i) :
    if (i == 0):
        return "negative"
    elif (i == 1):
        return "somewhat negative"
    elif (i == 2):
        return "neutral"
    elif (i == 3):
        return "somewhat positive"
    elif (i == 4):
        return "positive"
    else:
        return "unknown sentiment"

def getSentimentInt(s) :
    if (s is "negative"):
        return 0
    elif (s is "somewhat negative"):
        return 1
    elif (s is "neutral"):
        return 2
    elif (s is "somewhat positive"):
        return 3
    elif (s is "positive"):
        return 4
    else: 
        return -1
#https://spacy.io/usage/training#training-simple-style
#format of TRAIN_DATA = [  ("sentence 1", {'entities': [(0, 4, 'ORG')]}),
#  ("sentence 2", {'entities': [(0, 6, "ORG")]})]
#sample training data
#data=('I actually really like ... incarnations.', {'cats': {'POSITIVE': True}})
cats = []
i = 0
for y in train_cats:
    valueDict = {}
    for j in 0,1,2,3,4:
        if (y == j):
            valueDict[getSentimentStr(j)]= True
        else:
            valueDict[getSentimentStr(j)]= False
    dict1 = { 'cats' : valueDict}
    cats.append(dict1)
    i = i +1

train_data = list(zip(train_texts,cats))
# print one sample to see if everything is ok
print("text0=",train_data[0])
print("text1=",train_data[1])
print("text2=",train_data[2])
print("text3=",train_data[3])
print(nlp_en.pipe_names)
# add the text classifier to the pipeline if it doesn't exist
textcat = nlp_en.create_pipe('textcat')

nlp_en.add_pipe(textcat, last=True)

# add label to text classifier
for i in [getSentimentStr(0), getSentimentStr(1), getSentimentStr(2),getSentimentStr(3),getSentimentStr(4)]:
    textcat.add_label(i)

print(nlp_en.pipe_names)
from __future__ import unicode_literals, print_function
from spacy.util import minibatch, compounding

NUM_OF_ITERATIONS=5

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp_en.pipe_names if pipe != 'textcat']
with nlp_en.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp_en.begin_training()
    print("Training the model...")
    #print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
    print('{:^5}\t'.format('LOSS'))
    
    for i in range(NUM_OF_ITERATIONS):
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            #print(type(texts)) # class tuple
            #print(type(annotations)) # class tuple
            nlp_en.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            
        #with textcat.model.use_params(optimizer.averages):
        # evaluate on the dev data split off in load_data()
            #scores = evaluate(nlp_en.tokenizer, textcat, val_texts, val_cats)
            #print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                #.format(losses['textcat'], scores['textcat_p'], scores['textcat_r'], scores['textcat_f']))
        # print losses
        print('{0:.3f}'.format(losses['textcat']))
from pathlib import Path
output_dir="/tmp/"
output_dir = Path(output_dir)
nlp_en.to_disk(output_dir)
print("Saved model to", output_dir)
# test the trained model
test_text = "This movie sucked"
doc = nlp_en(test_text)
print(test_text, doc.cats)
print(test_text, sorted(doc.cats.items(), key=lambda val: val[1], reverse=True))
    
# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
doc2 = nlp2(test_text)
print(test_text, doc2.cats)
print(test_text, sorted(doc2.cats.items(), key=lambda val: val[1], reverse=True))
rawtestdf.describe()
#print(type(doc2.cats)) # dict
mydict = doc2.cats
for key, value in sorted(mydict.items(),  key=lambda val: val[1], reverse=True):
    print ("%s: %s" % (key, value))
# test 
import csv
with open('output.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['PhraseId', 'Sentiment'])
    for index, row in rawtestdf.iterrows():
        phraseId = row['PhraseId']
        text = row['Phrase']
        doc = nlp_en(text)
        for key, value in sorted(doc.cats.items(),  key=lambda val: val[1], reverse=True):
            writer.writerow([phraseId, getSentimentInt(key)])
            break
from itertools import islice
with open('output.csv') as myfile:
    head = list(islice(myfile, 10))
print(head)