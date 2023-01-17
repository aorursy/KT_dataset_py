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
import nltk

from collections import Counter

import itertools
import torch
class InputFeatures(object):

    """A single set of features of data."""



    def __init__(self, input_ids, label_id):

        self.input_ids = input_ids

        self.label_id = label_id
class Vocab:

    def __init__(self, itos, unk_index):

        self._itos = itos

        self._stoi = {word:i for i, word in enumerate(itos)}

        self._unk_index = unk_index

        

    def __len__(self):

        return len(self._itos)

    

    def word2id(self, word):

        idx = self._stoi.get(word)

        if idx is not None:

            return idx

        return self._unk_index

    

    def id2word(self, idx):

        return self._itos[idx]
from tqdm import tqdm_notebook
class TextToIdsTransformer:

    def transform():

        raise NotImplementedError()

        

    def fit_transform():

        raise NotImplementedError()



class SimpleTextTransformer(TextToIdsTransformer):

    def __init__(self, max_vocab_size):

        self.special_words = ['<PAD>', '</UNK>', '<S>', '</S>']

        self.unk_index = 1

        self.pad_index = 0

        self.vocab = None

        self.max_vocab_size = max_vocab_size

        

    def tokenize(self, text):

        return nltk.tokenize.word_tokenize(text.lower())

        

    def build_vocab(self, tokens):

        itos = []

        itos.extend(self.special_words)

        

        token_counts = Counter(tokens)

        for word, _ in token_counts.most_common(self.max_vocab_size - len(self.special_words)):

            itos.append(word)

            

        self.vocab = Vocab(itos, self.unk_index)

    

    def transform(self, texts):

        result = []

        for text in texts:

            tokens = ['<S>'] + self.tokenize(text) + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result

    

    def fit_transform(self, texts):

        result = []

        tokenized_texts = [self.tokenize(text) for text in texts]

        self.build_vocab(itertools.chain(*tokenized_texts))

        for tokens in tokenized_texts:

            tokens = ['<S>'] + tokens + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result
def build_features(token_ids, label, max_seq_len, pad_index, label_encoding):

    if len(token_ids) >= max_seq_len:

        ids = token_ids[:max_seq_len]

    else:

        ids = token_ids + [pad_index for _ in range(max_seq_len - len(token_ids))]

    return InputFeatures(ids, label_encoding[label])

        
def features_to_tensor(list_of_features):

    text_tensor = torch.tensor([example.input_ids for example in list_of_features], dtype=torch.long)

    labels_tensor = torch.tensor([example.label_id for example in list_of_features], dtype=torch.long)

    return text_tensor, labels_tensor
from sklearn import model_selection
imdb_df = pd.read_csv('../input/imdb_master.csv', encoding='latin-1')

dev_df = imdb_df[(imdb_df.type == 'train') & (imdb_df.label != 'unsup')]

test_df = imdb_df[(imdb_df.type == 'test')]

train_df, val_df = model_selection.train_test_split(dev_df, test_size=0.05, stratify=dev_df.label)
max_seq_len=200

classes = {'neg': 0, 'pos' : 1}
text2id = SimpleTextTransformer(10000)



train_ids = text2id.fit_transform(train_df['review'])

val_ids = text2id.transform(val_df['review'])

test_ids = text2id.transform(test_df['review'])
print(train_df.review.iloc[0][:160])

print(train_ids[0][:30])
train_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(train_ids, train_df['label'])]



val_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(val_ids, val_df['label'])]



test_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(test_ids, test_df['label'])]
print(train_features[3].input_ids)
train_tensor, train_labels = features_to_tensor(train_features)

val_tensor, val_labels = features_to_tensor(val_features)

test_tensor, test_labels = features_to_tensor(test_features)
print(train_tensor.size())
print(len(text2id.vocab))