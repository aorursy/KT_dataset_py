# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import operator 

import re

import gc

import keras

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
df = pd.concat([train, test])
print("Total number of examples: ", df.shape[0])
!pip install pytorch-pretrained-bert
import torch

from pytorch_pretrained_bert import BertTokenizer



# Load pre-trained model tokenizer (vocabulary)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
with open("vocabulary.txt", 'w') as f:

    

    # For each token...

    for token in tokenizer.vocab.keys():

        

        # Write it out and escape any unicode characters.            

        f.write(token + '\n')
# this buids the vocab. of our dataset

def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
vocab = build_vocab(df['text'])
list(vocab.keys())[:10]
# this check how much of our vocab is similar to the BERT vocab.

def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
print("BERT")

oov_bert = check_coverage(vocab, tokenizer.vocab)
oov_bert[:10]
tokenizer.vocab["I"]
tokenizer.vocab["i"]
df['lowered_text'] = df['text'].apply(lambda x: x.lower())
vocab_lower = build_vocab(df['lowered_text'])

print("BERT EMBEDDINGS")

oov_bert = check_coverage(vocab_lower, tokenizer.vocab)
oov_bert[:25]
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known
print("- Known Contractions -")

print("   BERT :")

print(known_contractions(tokenizer.vocab))
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
df['treated_text'] = df['lowered_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
vocab = build_vocab(df['treated_text'])

print("BERT : ")

oov_bert = check_coverage(vocab, tokenizer.vocab)
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown
print("BERT :")

print(unknown_punct(tokenizer.vocab, punct))
punct_mapping = { 'à': 'a'}
oov_bert[:10]
tokenizer.vocab["amp"]
print(tokenizer.tokenize("thunderstorm"))

print(tokenizer.tokenize("11-year-old"))

print(tokenizer.tokenize("@youtube"))

print(tokenizer.tokenize("\x89û_"))
bad_words = []

for i in range(len(oov_bert)):

    if oov_bert[i][0][0] =="\x89":

        bad_words.append(oov_bert[i])
bad_dict = {}

for i in range(len(bad_words)):

    bad_dict[bad_words[i][0]] = ""
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = bad_dict  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
df['treated_text'] = df['treated_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
vocab = build_vocab(df['treated_text'])

print("BERT : ")

oov_bert = check_coverage(vocab, tokenizer.vocab)
oov_bert[:25]
for i in oov_bert[:25]:

    print(tokenizer.tokenize(i[0]))
explicit_mapping = {"\x89û": "", "mh370" : "flight", "legionnaires": "pneumonia", 

                   "derailment": "railway accident", "inundated": "flood", "deluged": "flood", 

                   "curfew": "stay at home","obliteration": "destruction", 

                   "quarantine": "prevent the spread of disease", "lol": "laugh", 

                   "obliterate": "destroy", "hijacking": "seize", "detonation": "explosion", 

                   "electrocuted": "killed", "destroyd": "destroyed"}
def explicit_changes(text, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    return text
df['treated_text'] = df['treated_text'].apply(lambda x: explicit_changes(x, explicit_mapping))
vocab = build_vocab(df['treated_text'])

print("BERT : ")

oov_bert = check_coverage(vocab, tokenizer.vocab)
oov_bert[:20]
# lower

train['treated_text'] = train['text'].apply(lambda x: x.lower())

# clean contractions

train['treated_text'] = train['treated_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

# clean special chars - this is optional as most of the punct. are in BERT embed.

train['treated_text'] = train['treated_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# cleaning some word

train['treated_text'] = train['treated_text'].apply(lambda x: explicit_changes(x, explicit_mapping))
train.head()
test['treated_text'] = test['text'].apply(lambda x: x.lower())

test['treated_text'] = test['treated_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

test['treated_text'] = test['treated_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test['treated_text'] = test['treated_text'].apply(lambda x: explicit_changes(x, explicit_mapping))
test.head()
# Saving out work

train.to_csv("train_BERT_preprocessed.csv")

test.to_csv("test_BERT_preprocessed.csv")