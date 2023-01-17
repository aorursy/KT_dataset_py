# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

import string

import gensim

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import re

import matplotlib.pyplot as plt
metadata  = []

with open("../input/arxiv/arxiv-metadata-oai-snapshot.json", 'r') as f:

    for line in f: 

        metadata.append(json.loads(line))
# for testing 

metadata = pd.DataFrame(metadata)

metadata = pd.concat([metadata[:250000],  

                      metadata[metadata['id'].isin(['math-ph/0409039','math-ph/0408005','math-ph/0212014'])]])

print(metadata.shape)

metadata.head()
keep_cols = ['id', 'title', 'abstract', 'categories']

metadata = metadata[keep_cols]

print(metadata.shape)
######## ADD PROCESSING FOR NEW IDS #####################

## requires internet to be enabled

#id_ = '0704.0001' # test



# #def pull_content(id):

#f = urllib.request.urlopen('https://arxiv.org/pdf/{}.pdf'.format(id)).read()

#f = StringIO(f)

#reader = PyPDF2.PdfFileReader(f)

# #return content
unique_categories = set([i for l in [x.split(' ') for x in metadata['categories']] for i in l])

print(len(unique_categories))



# create label column for each labeled category

for un in unique_categories:

    metadata[un] = np.where(metadata['categories'].str.contains(un), 1, 0)
### APPLIES TO ALL NLP APPLICATIONS ###

# remove duplicate records which contain different flags

metadata = metadata.drop(columns = 'categories').groupby(by = ['id', 'title', 'abstract'],as_index = False).max()

# remove abstracts from withdrawn records

metadata = metadata[metadata['abstract'].str.contains('paper has been withdrawn') == False]

# lower abstract and remove numbers, punctuation, and special characters

#metadata['abstract'] = [a.strip() for a in metadata['abstract']]

metadata['abstract'] = [a.lower().strip() for a in metadata['abstract']]

metadata['abstract'] = metadata['abstract'].str.replace('\n', ' ', regex = False).str.replace(r'\s\s+', ' ', regex = True)

metadata['abstract'] = metadata['abstract'].str.replace('([.,!?()])', r' \1 ')
metadata['abstract'][0]
# ### PREPROCESSING FUNCTIONS ###

# # lemmatizing function

# def lemmatize_text(text_string, lemmatizer):

#     word_list = nltk.word_tokenize(text_string)

#     # Lemmatize list of words and join

#     lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])

#     return lemmatized_output

# ### PREPROCESSING FOR EMBEDDING AND SKLEARN VECTORIZERS ###

# metadata['abstract_no_punct'] = metadata['abstract'].str.replace('-', '', regex = False).str.replace(r'\n', ' ', regex = False).str.replace(r'[^a-z ]+', ' ', regex = True).str.replace(r'\s\s+', ' ', regex = True)

# # lemmatize

# lemmatizer = WordNetLemmatizer()

# metadata['abstract_lemmatized'] = [lemmatize_text(ab, lemmatizer) for ab in metadata['abstract']]

# # generate formatted stop words + single letters and spelled numbers (expand as necesary)

# stopwords_nltk = set([re.sub( r'[^a-z ]+', '',s) for s in stopwords.words('english')] + list(string.ascii_lowercase) + ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])

# # exclude stopwords

# metadata['abstract_lemmatized_no_stopwords'] = metadata['abstract_lemmatized'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords_nltk]))
# stopwords_nltk
# # generate topic list

# keep_cols = [c for c in metadata.columns if c not in (list(metadata.columns[:3]) + ['abstract_lemmatized','abstract_lemmatized_no_stopwords'])]

# topic_list = list(metadata[keep_cols].sum().sort_values(ascending=False)[:15].index)
# #%matplotlib inline

# def create_plots_word_freq(topic_list):

#     for topic in topic_list:

#         #print(topic)

#         # aggregate all text and find frequent words

#         all_tagged_text = ' '.join(metadata[metadata[topic]]['abstract_lemmatized_no_stopwords'])

#         freq_dist = nltk.FreqDist(all_tagged_text.split())

#         # plotting

#         fig = plt.figure(figsize=(16, 9))

#         ax = fig.gca() # Get current axes. This is the Axes object that is created by default when we make our figure

#         fig.suptitle('50 Most Common Words in Abstract for ' + topic.title() + ' Papers', fontsize=20)

#         plt.xlabel('Word', fontsize=14)

#         plt.ylabel('Frequency', fontsize=14)

#         plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

#         plt.bar(list(map(list, zip(*freq_dist.most_common(50))))[0], list(map(list, zip(*freq_dist.most_common(50))))[1])

#         plt.show()

# create_plots_word_freq(topic_list)
category_count = metadata[metadata.columns[3:].to_list()].sum()

labels = category_count.sort_values(ascending = False).index.to_list()

counts = category_count.sort_values(ascending = False).values



fig = plt.figure(figsize=(16, 9))

ax = fig.gca()

fig.suptitle('Top 50 Categories', fontsize=20)

plt.ylabel('Number of Papers', fontsize=14)

plt.xlabel('Labeled Paper Category', fontsize=14)

plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

plt.bar(labels[:50],counts[:50])

plt.show()



fig = plt.figure(figsize=(16, 9))

ax = fig.gca()

fig.suptitle('Top 10 Categories', fontsize=20)

plt.ylabel('Number of Papers', fontsize=14)

plt.xlabel('Labeled Paper Category', fontsize=14)

plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

plt.bar(labels[:10],counts[:10])

plt.show()



fig = plt.figure(figsize=(16, 9))

ax = fig.gca()

fig.suptitle('Bottom 50 Categories', fontsize=20)

plt.ylabel('Number of Papers', fontsize=14)

plt.xlabel('Labeled Paper Category', fontsize=14)

plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

plt.bar(labels[105:],counts[105:])

plt.show()





fig = plt.figure(figsize=(16, 9))

ax = fig.gca()

fig.suptitle('Bottom 10 Categories', fontsize=20)

plt.ylabel('Number of Papers', fontsize=14)

plt.xlabel('Labeled Paper Category', fontsize=14)

plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

plt.bar(labels[145:],counts[145:])

plt.show()
row_sums = metadata[metadata.columns[3:].to_list()].sum(axis = 1)

multilabel_counts = row_sums.value_counts()
multilabel_counts
fig = plt.figure(figsize=(6, 4))

ax = fig.gca()

fig.suptitle('Multi-Label Counts 1-6', fontsize=20)

plt.ylabel('Number of Papers', fontsize=14)

plt.xlabel('Number of Categories', fontsize=14)

plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

ax.set_xticks(list(range(1,14)))

plt.bar(multilabel_counts.index[:6],multilabel_counts[:6].values)

plt.show()



fig = plt.figure(figsize=(6, 4))

ax = fig.gca()

fig.suptitle('Multi-Label Counts > 6', fontsize=20)

plt.ylabel('Number of Papers', fontsize=14)

plt.xlabel('Number of Categories', fontsize=14)

plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

ax.set_xticks(list(range(1,14)))

plt.bar(multilabel_counts.index[7:],multilabel_counts[7:].values)

plt.show()
sentence_lengths = [len(t.split()) for t in metadata['abstract']]

sentence_lengths.sort()

plt.hist(sentence_lengths)

plt.show()
plt.hist([i for i in sentence_lengths if i > 299 and i < 400])

plt.show()
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification, AdamW

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torch.utils.data import Dataset, TensorDataset, DataLoader

from sklearn.metrics import (

    accuracy_score, 

    f1_score, 

    classification_report

)
exclude_categories = category_count.sort_values().index[:10].to_list()

categories = [c for c in unique_categories if c not in exclude_categories]

len(categories)

# remove columns pertaining to categories to exclude  

metadata = metadata[list(metadata.columns[:3]) + categories]
# fixed parameters and hyperparameters for dataset creation and model training

class Config:

    def __init__(self, categories):

        # allow class to be inherited

        super(Config, self).__init__()

        

        # general parameters

        self.SEED = 9

        self.MODEL_PATH = "allenai/scibert_scivocab_uncased"

        self.NUM_LABELS = len(categories)

        

        # load tokenizer and set related parameters

        self.TOKENIZER = BertTokenizer.from_pretrained(self.MODEL_PATH)

        self.MAX_LENGTH = 330 # from EDA

        

        # determine optimal batch size based on 

        self.N_GPU = torch.cuda.device_count()

        if self.N_GPU == 0:

            self.N_GPU = 1

        self.BATCH_SIZE = self.N_GPU * 8

            

        # validation & test split

        self.VALIDATION_SPLIT = .3

        

        # set model parameters

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.FULL_FINETUNING = True

        self.LR = 3e-5

        self.CRITERION = nn.BCEWithLogitsLoss()

        self.SAVE_BEST_ONLY = True

        self.N_VALIDATE_DUR_TRAIN = 3

        self.EPOCHS = 4



config = Config(categories)
# isolate data

labels = metadata.loc[:, categories].values

input_data = metadata[['id', 'abstract']]
# get rid of the old objects

del stopwords, metadata, keep_cols, unique_categories, un, category_count, counts, fig, ax, row_sums, multilabel_counts, sentence_lengths, exclude_categories
%%time

tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)

tokenized_abstracts = tokenizer.batch_encode_plus(

            input_data['abstract'],

            max_length = config.MAX_LENGTH,

            pad_to_max_length = True,

            truncation = True,

            return_attention_mask = True,

            return_token_type_ids = False,

            return_tensors = 'pt'

        )
print(input_data['abstract'].shape)

print(tokenized_abstracts['input_ids'].shape)

print(tokenized_abstracts['attention_mask'].shape)

print(labels.shape)
# initial train and test split

token_train, token_test, mask_train, mask_test, \

y_train, y_test = train_test_split(np.array(tokenized_abstracts['input_ids']),

                                   np.array(tokenized_abstracts['attention_mask']), 

                                   np.array(labels), 

                                   test_size = config.VALIDATION_SPLIT,

                                   random_state = config.SEED)

# split test into test and validation

token_val, token_test, mask_val, mask_test, \

y_val, y_test = train_test_split(token_test,

                                 mask_test,

                                 y_test,

                                 test_size = 0.5,

                                 random_state = config.SEED)
print('ABSTRACT - TRAIN DIM:', token_train.shape)

print('MASK - TRAIN DIM:', mask_train.shape)

print('LABEL - TRAIN DIM:', y_train.shape)



print('ABSTRACT - TEST DIM:', token_test.shape)

print('MASK - TEST DIM:', mask_test.shape)

print('LABEL - TEST DIM:', y_test.shape)



print('ABSTRACT - VAL DIM:', token_val.shape)

print('MASK - VAL DIM:', mask_val.shape)

print('LABEL - VAL DIM:', y_val.shape)
# def generate_dataloader(token_ids, attention_mask, labels, batch_size = 8):

#     dataset = TensorDataset(token_ids.long(),

#                             attention_mask.long(),

#                             labels.long())

    

#     dataloader = DataLoader(dataset, batch_size = batch_size)

#     return dataloader
# # Create the dataloaders

# train_dataloader = generate_dataloader(token_train, mask_train, y_train, batch_size = config.BATCH_SIZE)

# valid_dataloader = generate_dataloader(token_val, mask_val, y_val, batch_size = config.BATCH_SIZE)

# test_dataloader = generate_dataloader(token_test, mask_test, y_test, batch_size = config.BATCH_SIZE)
# class arxiv_dataset(torch.utils.data.Dataset):

#     def __init__(self, abstrcts, lbls, msks):

#         self.abstrcts = abstrcts

#         self.lbls = lbls

#         self.msks = msks

        

#     def __len__(self):

#         return self.lbls.shape[0]

    

#     def __getitem__(self, index):

#         abstracts_ = self.abstrcts[index, :]

#         labels_ = self.lbls[index, :]

#         masks_ = self.msks[index, :]

#         return abstracts_, labels_, masks_
# ### np.array()

# # data

# train_data = arxiv_dataset(token_train, y_train, mask_train)

# val_data = arxiv_dataset(token_val, y_val, mask_val)

# test_data = arxiv_dataset(token_test, y_test, mask_test)
class arxiv_dataset(torch.utils.data.Dataset):

            

    def __init__(self, abstrcts, lbls, msks):

        self.abstrcts = torch.Tensor(abstrcts).long()

        self.msks = torch.Tensor(msks).long()

        self.lbls = torch.Tensor(lbls).float()

        

    def __len__(self):

        return self.lbls.shape[0]

    

    def __getitem__(self, index):

        abstracts_ = self.abstrcts[index, :]

        labels_ = self.lbls[index, :]

        masks_ = self.msks[index, :]

        return abstracts_, labels_, masks_
### tensors

train_data = arxiv_dataset(token_train, y_train, mask_train)

val_data = arxiv_dataset(token_val, y_val, mask_val)

test_data = arxiv_dataset(token_test, y_test, mask_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE)

val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE)
model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=config.NUM_LABELS)

if torch.cuda.is_available():

    model = model.cuda()  



## set model

# class SciBert(nn.Module):

#     def __init__(self):

#         super(SciBert, self).__init__()

        

#         # NN Architecture

#         self.bert = BertForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=config.NUM_LABELS)

        

#         def forward(self, tokens):

#             _,output = self.bert(tokens)

#             logits = output.logits

#             return logits

# model = SciBert()

# if torch.cuda.is_available():

#     model = model.cuda()
# set set parameters

param_optimizer = list(model.named_parameters())



# According to the huggingface recommendations

# weight decay is set to 0 for bias layers

no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

                                 'weight_decay_rate': 0.01},

                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

                                 'weight_decay_rate': 0.0}]



# Using BERT's Adam optimizer similar to

# the original Tensorflow optimizer

optimizer = AdamW(optimizer_grouped_parameters,

                  lr = config.LR,

                  weight_decay = 0.01,

                  correct_bias = False)



criterion = nn.BCEWithLogitsLoss()
%%time

if config.N_GPU > 1:

    model = nn.DataParallel(model)

        

epoch_train_loss = []

epoch_valid_loss = []



for epoch in range(config.EPOCHS):

    print('EPOCH:', epoch)



    train_loss = 0.0

    valid_loss = 0.0



    ######### TRAINING #############

    # set model to train mode

    model.train()



    batch = 1

    # iterate through each observation

    for data in train_dataloader:

        print('BATCH:', batch)

        abstracts_, labels_, masks_ = data



        # move data to GPU

        if torch.cuda.is_available():

            abstracts_ = abstracts_.cuda()

            masks_ = masks_.cuda()

            labels_ = labels_.cuda()



        # zero out optimizer gradients

        optimizer.zero_grad()



        # fit model and calculate loss

        logits = model(input_ids = abstracts_, attention_mask = masks_)[0]

        loss = criterion(logits, labels_)



        if config.N_GPU > 1 :

            loss = loss.mean()



        loss.backward()

        optimizer.step()



        train_loss += loss.item()

        print('batch loss:', loss.item())

        batch += 1

        

    print(f"Train loss:\t{train_loss/len(train_dataloader)}")



    ###### VALIDATION ########

    # set model to train mode

    model.eval()



    valid_truth = []

    valid_preds = []

    

    batch = 1

    # iterate through each observation

    for data in val_dataloader:

        print('BATCH:', batch)

        abstracts_, labels_, masks_ = data

        # move data to GPU

        if torch.cuda.is_available():

            abstracts_ = abstracts_.cuda()

            masks_ = masks_.cuda()

            labels_ = labels_.cuda()



        # zero out optimizer gradients

        optimizer.zero_grad()



        # fit model and calculate loss

        logits = model(input_ids = abstracts_, attention_mask = masks_)[0]

        loss = criterion(logits, labels_)



        if config.N_GPU > 1 :

            loss = loss.mean()



        valid_loss += loss.item()

        batch += 1



        # keep list of outputs for validation

        valid_truth.extend(labels_.cpu().numpy())

        valid_preds.extend(logits.cpu().detach().numpy())



    print(f"Valid loss:\t{valid_loss/len(valid_dataloader)}")



    # If validation loss improved,

    # set the best model to the current model

    if len(epoch_valid_loss) == 0:

        best_model = model

    else:

        if epoch_valid_loss[-1] > valid_loss:

            best_model = model



    # update epoch loss lists 

    epoch_train_loss.extend(train_loss/len(train_dataloader))

    epoch_valid_loss.extend(valid_loss/len(valid_dataloader))
torch.save(best_model.state_dict(), '10_19_0930.pt')
count_vectorizer = CountVectorizer(stop_words = stopwords_nltk , ngram_range = (1,2)

                                   ,max_df = .99, min_df = .02)

X_cv = count_vectorizer.fit_transform(metadata['abstract_lemmatized']) 



tfidf_vectorizer = TfidfVectorizer(stop_words = stopwords_nltk , ngram_range = (1,2)

                                   ,max_df = .99, min_df = .02)

X_tf = tfidf_vectorizer.fit_transform(metadata['abstract_lemmatized']) 
## train word embedding model using word2vec model

input_data = metadata['abstract_lemmatized_no_stopwords'].apply(lambda x: x.split())

arxiv_w2v_model = gensim.models.Word2Vec(input_data)

arxiv_w2v_model['algorithmic']
# gensim api contains sample corpora and pretrained models (make sure internet is on)

import gensim.downloader as api

info = api.info()

for model_name, model_data in sorted(info['models'].items()):

    print(

        '%s (%d records): %s' % (

            model_name,

            model_data.get('num_records', -1),

            model_data['description'][:40] + '...',

        )

    )
# use pretrained w2v model (make sure internet is on)

pretrained_w2v_model = api.load("glove-wiki-gigaword-100")

pretrained_w2v_model['glass']
def create_abstract_vectors(abstract, model, embedding_length):

    word_vector_sum = [0]*embedding_length ## using embeddings of length 50

    no_embedding_word_list = []

    for word in abstract.split():

        try:

            word_vector = model[word]

            word_vector_sum = [sum(x) for x in zip(word_vector_sum, word_vector)]

        except:

            #print(word, 'does not have embedding')

            if word not in no_embedding_word_list:

                no_embedding_word_list.append(word)

            pass

    if len(no_embedding_word_list) > 0:

        print(' '.join(no_embedding_word_list))

    return [added_elt/len(abstract.split()) for added_elt in word_vector_sum]



embedding_length = 100 # pretrained_w2v_model, arxiv_w2v_model

embeddings_pretrained = metadata['abstract_lemmatized_no_stopwords'].apply(lambda x: create_abstract_vectors(x, pretrained_w2v_model, embedding_length))

embeddings_arxiv = metadata['abstract_lemmatized_no_stopwords'].apply(lambda x: create_abstract_vectors(x, arxiv_w2v_model, embedding_length))
# import modules

from keras.models import Model

from keras.layers import Input, LSTM, Dense



# Define an input sequence and process it.

encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder = LSTM(latent_dim, return_state=True)

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.

encoder_states = [state_h, state_c]



# Set up the decoder, using `encoder_states` as initial state.

decoder_inputs = Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,

# and to return internal states as well. We don't use the 

# return states in the training model, but we will use them in inference.

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs,

                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)



# Define the model that will turn

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Run training

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

          batch_size=batch_size,

          epochs=epochs,

          validation_split=0.2)
encoder_model = Model(encoder_inputs, encoder_states)



decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(

    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(

    [decoder_inputs] + decoder_states_inputs,

    [decoder_outputs] + decoder_states)
def decode_sequence(input_seq):

    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)



    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Populate the first character of target sequence with the start character.

    target_seq[0, 0, target_token_index['\t']] = 1.



    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).

    stop_condition = False

    decoded_sentence = ''

    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(

            [target_seq] + states_value)



        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = reverse_target_char_index[sampled_token_index]

        decoded_sentence += sampled_char



        # Exit condition: either hit max length

        # or find stop character.

        if (sampled_char == '\n' or

           len(decoded_sentence) > max_decoder_seq_length):

            stop_condition = True



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1, 1, num_decoder_tokens))

        target_seq[0, 0, sampled_token_index] = 1.



        # Update states

        states_value = [h, c]



    return decoded_sentence