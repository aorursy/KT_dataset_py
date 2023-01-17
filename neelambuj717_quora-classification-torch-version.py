import pandas as pd

import numpy as np

import nltk

from sklearn.metrics import f1_score

import re

import string

import xgboost as xgb

import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold, GridSearchCV, RandomizedSearchCV

import catboost as cat

import pandas as pd

import numpy as np

import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping

from nltk import word_tokenize

from nltk.corpus import stopwords

# stop_words = stopwords.words('english')



import warnings

from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore")

eng_stopwords = set(stopwords.words("english"))

import os



####################

import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
os.listdir('../input')
pd.read_csv('../input/quora-insincere-questions-classification/train.csv', nrows = 100).head()
# We need to define fields for how the every column will be processed in the tabular dataset

from torchtext.data import Field, TabularDataset, BucketIterator

import torchtext

import torch

import torch.nn as nn
%%time

# Ensure there is a field expression for every column in the data

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))



# Fields are ready for being used in the tabular dataset 

tokenize = lambda x: x.split()

question = Field(sequential=True, use_vocab=True,tokenize=word_tokenize, stop_words = stop_words)

target = Field(sequential=False, use_vocab=False, is_target = True, dtype=torch.float64)

qid = torchtext.data.Field(use_vocab=False, sequential=False)





train_set = TabularDataset(path = '../input/quora-insincere-questions-classification/train.csv',

                      format='csv',

                      fields = [('qid', None), ('question_text', question), ('target', target)],

                          skip_header=True)

test_set = TabularDataset(path = '../input/quora-insincere-questions-classification/test.csv',

                      format='csv',

                      fields = [('qid', qid), ('question_text', question)],

                          skip_header=True)



print(train_set[0].__dict__.keys())

train_set[1].question_text, train_set[1].target



# Building vocabulary using glove pretrained vectors

question.build_vocab(train_set, min_freq = 3, max_size = 2000000)

question.vocab.load_vectors(torchtext.vocab.Vectors('../input/glove6b/glove.6B.300d.txt'))

print(question.vocab.vectors.shape)





import random

train_data, valid_data = train_set.split(split_ratio = 0.8, random_state = random.seed(123))

train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),

                                        batch_size = 128,

                                       sort_key = lambda x: len(x.question_text),

                                       sort_within_batch = True,

                                        device='cuda')



test_iter = BucketIterator(test_set,

                          batch_size = 128,

                          sort = False, sort_within_batch=False,

                          device='cuda')
# Testing the iterator for train and valid data

for batch in train_iter:

    print(batch.question_text.shape)

    break
class params():

    input_dim = question.vocab.vectors

    batch_size = 128

    embedding_dim = 300

    hidden_dim = 128

    output_dim = 1

    learning_rate = 1e-3

    num_layers = 3

    bidirectional =True

    dropout_prob = 0.2

    padding_idx = question.vocab.stoi[question.pad_token]

    static=False

    device='cuda'

    

args = params()
import torch.nn as nn

class BiLSTMS(nn.Module):

    def __init__(self,vocab_size,  embedding_dim, static, hidden_dim, output_dim, padding_idx, num_layers,

        bidirectional, dropout_prob):

        super(BiLSTMS, self).__init__()

        

        # Initializing the embedding layer for the network

        self.embedding = nn.Embedding.from_pretrained(vocab_size, embedding_dim, 

                                                      padding_idx=padding_idx)

        self.static = static

        # Making embeddings trainable 

        if self.static:

            self.embedding.weight.requires_grad = False

            

        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers,

                    bidirectional= bidirectional, dropout=dropout_prob)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout_prob)



    def forward(self, text):

        # Text input  dimensions = [sentence_len, batch_size]

        embedding = self.embedding(text)

        embedding = self.dropout(embedding)

        

        # Removed the neurons who have a probability < dropout_prob



        # Padding the sequences beyond the max sequence length for prediction of sent.

#         print(text.shape[0])

#         packed_embedding = nn.utils.rnn.pack_padded_sequence(embedding, text.shape[0])



        packed_output, (hidden_state, cell_state) = self.LSTM(embedding)

#         print(hidden_state.shape, cell_state.shape)

        

        # Unpack the sequences using

#         output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # Concating the final forward and backward hidden state to make the predictions 

        # for the sentiment 



        hidden_final = self.dropout(torch.cat((hidden_state[-2, : , : ], hidden_state[-1, :, : ]), dim = 1))



        return self.fc(hidden_final)
# args.static
import torch.optim as optim

from sklearn.metrics import f1_score, accuracy_score

from tqdm import tqdm_notebook, tqdm



# Model Instantiation

model = BiLSTMS(args.input_dim, args.embedding_dim, args.static, args.hidden_dim, args.output_dim, 

                args.padding_idx, args.num_layers, args.bidirectional, args.dropout_prob)

model = model.to(args.device)





optimizer = optim.Adam(model.parameters())

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode ='min', factor = 0.2,

                                                patience=4, verbose=True)

criterion = nn.BCEWithLogitsLoss()

criterion = criterion.to(args.device)
def evaluate(model, iterator, criterion):

    targets= []

    preds = []

    with torch.no_grad():

        loop = tqdm(enumerate(iterator), position = 0, leave =True)

        for i, batch in loop:

            text = batch.question_text

            predictions = model(text).squeeze(1).double()

            actual_labels = batch.target

            targets += actual_labels.to('cpu').numpy().tolist()

            preds += predictions.to('cpu').numpy().tolist()

            

            loop.set_description(f"Evaluating model performance on validation dataset")

    print(len(preds), len(targets))

    

    label = [1 if pred >= 0.80 else 0 for pred in preds]

    return f1_score(targets, label), accuracy_score(targets, label)
def count_params(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters in the Bi-LSTM model are {count_params(model)}")



# pretrained_embedding_vec = question.vocab.vectors

# pretrained_embedding_vec.shape



# # Changing the default Embeddings in the model to pretrained embeddings

# model.embedding.weight.data.copy_(pretrained_embedding_vec)



# # Getting UNK index from the vocab for question text

# UNK_idx = question.vocab.stoi[question.unk_token]

# model.embedding.weight.data[UNK_idx] = torch.zeros(args.embedding_dim)

# model.embedding.weight.data[args.padding_idx] = torch.zeros(args.embedding_dim)

# print(model.embedding.weight.data)

# ## Training Loop for the LSTM model
f1_score_list = [0]

accuracy_list = [0]

num_epochs = 15





for epoch in range(num_epochs):

    epoch_loss = 0

    no_improve_count = 0

        

    loop = tqdm(enumerate(train_iter),  position = 0, leave=True)

    for i, batch in loop:

        optimizer.zero_grad()

        text = batch.question_text

        model.train()

        predictions = model(text).squeeze(1).double()

        loss = criterion(batch.target, predictions)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        loop.set_description(f"Epoch {epoch +1}/{num_epochs}")

        loop.set_postfix(loss = loss.item(), acc = accuracy_list[epoch], f1_score = f1_score_list[epoch])



    # Metrics on the evaluation set

    model.eval()

    f1, acc = evaluate(model, valid_iter, criterion)

    mean_loss = epoch_loss / len(train_iter)

    scheduler.step(mean_loss)

    

    f1_score_list.append(f1)

    accuracy_list.append(acc)
# f1, acc
# preds
# targets= []

# preds = []

# with torch.no_grad():

#     loop = tqdm(enumerate(valid_iter), position = 0, leave =True)

#     for i, batch in loop:

#         text = batch.question_text

#         predictions = model(text).squeeze(1).double()

#         actual_labels = batch.target

#         targets += actual_labels.to('cpu').numpy().tolist()

#         preds += predictions.to('cpu').numpy().tolist()



#         loop.set_description(f"Evaluating model performance on validation dataset")

# print(len(preds), len(targets))



# label = [1 if pred >= 0.80 else 0 for pred in preds]

# f1_score(targets, label), accuracy_score(targets, label)
np.sum(label)
# os.listdir('../input')



# train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv',)

# test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')



# train.shape, test.shape



# train['target'].value_counts()



# # Preprocessing the text



# train['len_of_sentence'] = train['question_text'].apply(lambda x: len(x.split()))

# import seaborn as sns

# sns.kdeplot(train['len_of_sentence'])

# plt.show()



# import matplotlib.pyplot as plt



# plt.figure(figsize = (15, 8))

# plt.title("Disbution of lengths of sentences")

# train['len_of_sentence'].value_counts().plot(kind='bar')

# plt.xticks(rotation = 45)

# plt.xlabel("Length of sentence")

# plt.ylabel("#Sentences")

# plt.show()



# puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

#  '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

#  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

#  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

#  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



# def clean_text(x):

#     x = str(x)

#     for punct in puncts:

#         x = x.replace(punct, f' {punct} ')

#     return x



# def clean_numbers(x):

#     x = re.sub('[0-9]{5,}', '#####', x)

#     x = re.sub('[0-9]{4}', '####', x)

#     x = re.sub('[0-9]{3}', '###', x)

#     x = re.sub('[0-9]{2}', '##', x)

#     return x



# mispell_dict = {"aren't" : "are not",

# "can't" : "cannot",

# "couldn't" : "could not",

# "didn't" : "did not",

# "doesn't" : "does not",

# "don't" : "do not",

# "hadn't" : "had not",

# "hasn't" : "has not",

# "haven't" : "have not",

# "he'd" : "he would",

# "he'll" : "he will",

# "he's" : "he is",

# "i'd" : "I would",

# "i'd" : "I had",

# "i'll" : "I will",

# "i'm" : "I am",

# "isn't" : "is not",

# "it's" : "it is",

# "it'll":"it will",

# "i've" : "I have",

# "let's" : "let us",

# "mightn't" : "might not",

# "mustn't" : "must not",

# "shan't" : "shall not",

# "she'd" : "she would",

# "she'll" : "she will",

# "she's" : "she is",

# "shouldn't" : "should not",

# "that's" : "that is",

# "there's" : "there is",

# "they'd" : "they would",

# "they'll" : "they will",

# "they're" : "they are",

# "they've" : "they have",

# "we'd" : "we would",

# "we're" : "we are",

# "weren't" : "were not",

# "we've" : "we have",

# "what'll" : "what will",

# "what're" : "what are",

# "what's" : "what is",

# "what've" : "what have",

# "where's" : "where is",

# "who'd" : "who would",

# "who'll" : "who will",

# "who're" : "who are",

# "who's" : "who is",

# "who've" : "who have",

# "won't" : "will not",

# "wouldn't" : "would not",

# "you'd" : "you would",

# "you'll" : "you will",

# "you're" : "you are",

# "you've" : "you have",

# "'re": " are",

# "wasn't": "was not",

# "we'll":" will",

# "didn't": "did not",

# "tryin'":"trying"}



# def _get_mispell(mispell_dict):

#     mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

#     return mispell_dict, mispell_re



# mispellings, mispellings_re = _get_mispell(mispell_dict)

# def replace_typical_misspell(text):

#     def replace(match):

#         return mispellings[match.group(0)]

#     return mispellings_re.sub(replace, text)



# # Clean the text

# train["question_text"] = train["question_text"].apply(lambda x: clean_text(x.lower()))

# test["question_text"] = test["question_text"].apply(lambda x: clean_text(x.lower()))



# # Clean numbers

# train["question_text"] = train["question_text"].apply(lambda x: clean_numbers(x))

# test["question_text"] = test["question_text"].apply(lambda x: clean_numbers(x))



# # Clean speelings

# train["question_text"] = train["question_text"].apply(lambda x: replace_typical_misspell(x))

# test["question_text"] = test["question_text"].apply(lambda x: replace_typical_misspell(x))



# # os.listdir('../input/glove840b300dtxt/glove.840B.300d.txt')



# train['length'] = train['question_text'].apply(lambda x: len(x.split()))

# test['length'] = test['question_text'].apply(lambda x: len(x.split()))



# np.mean(train['length']), np.mean(test['length']), np.max(train['length']), np.max(test['length'])



# from keras.preprocessing.text import Tokenizer

# from keras.preprocessing.sequence import pad_sequences



# num_words = 120000

# tx = Tokenizer(num_words = num_words, lower = True, filters = '')

# full_text = list(train['question_text'].values) + list(test['question_text'].values)

# tx.fit_on_texts(full_text)



# train_tokenized = tx.texts_to_sequences(train['question_text'].fillna('missing'))

# test_tokenized = tx.texts_to_sequences(test['question_text'].fillna('missing'))



# max_len = 100

# X_train = pad_sequences(train_tokenized, maxlen = max_len)

# X_test = pad_sequences(test_tokenized, maxlen = max_len)



# X_train.shape