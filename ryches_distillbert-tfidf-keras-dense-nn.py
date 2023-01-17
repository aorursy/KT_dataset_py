import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")
!pip install ../input/sacremoses/sacremoses-master/ > /dev/null
import sys

sys.path.insert(0, "../input/transformers/transformers-master/")
target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

import torch

from transformers import *

import tqdm

tokenizer = DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")

model = DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
model.cuda()
question_body_ids = train["question_body"].str.slice(0, 500).str.lower().apply(tokenizer.encode)

question_body_ids_test = test["question_body"].str.slice(0, 500).str.lower().apply(tokenizer.encode)

question_body_vectors = []

question_body_vectors_test = []

for question_body in tqdm.tqdm(question_body_ids):

    input_ids = torch.Tensor(question_body).to(torch.int64).unsqueeze(0)

    try:

        outputs = model(input_ids.cuda())

        question_body_vectors.append(outputs[0].detach().cpu().numpy().max(axis = 1))

    except:

        question_body_vectors.append(np.zeros(outputs[0].detach().cpu().numpy().max(axis = 1)).shape)

    

for question_body in tqdm.tqdm(question_body_ids_test):

    input_ids = torch.Tensor(question_body).to(torch.int64).unsqueeze(0)

    try:

        outputs = model(input_ids.cuda())

        question_body_vectors_test.append(outputs[0].detach().cpu().numpy().max(axis = 1))

    except:

        question_body_vectors_test.append(np.zeros(outputs[0].detach().cpu().numpy().max(axis = 1)).shape)
question_title_ids = train["question_title"].str.slice(0, 500).str.lower().apply(tokenizer.encode)

question_title_ids_test = test["question_title"].str.slice(0, 500).str.lower().apply(tokenizer.encode)

question_title_vectors = []

question_title_vectors_test = []

for question_title in tqdm.tqdm(question_title_ids):

    input_ids = torch.Tensor(question_title).to(torch.int64).unsqueeze(0)

    try:

        outputs = model(input_ids.cuda())

        question_title_vectors.append(outputs[0].detach().cpu().numpy().max(axis = 1))

    except:

        question_title_vectors.append(np.zeros(outputs[0].detach().cpu().numpy().max(axis = 1)).shape)

    

for question_title in tqdm.tqdm(question_title_ids_test):

    input_ids = torch.Tensor(question_title).to(torch.int64).unsqueeze(0)

    try:

        outputs = model(input_ids.cuda())

        question_title_vectors_test.append(outputs[0].detach().cpu().numpy().max(axis = 1))

    except:

        question_title_vectors_test.append(np.zeros(outputs[0].detach().cpu().numpy().max(axis = 1)).shape)
answer_ids = train["answer"].str.slice(0, 500).str.lower().apply(tokenizer.encode)

answer_ids_test = test["answer"].str.slice(0, 500).str.lower().apply(tokenizer.encode)

answer_vectors = []

answer_vectors_test = []

for answer in tqdm.tqdm(answer_ids):

    input_ids = torch.Tensor(answer).to(torch.int64).unsqueeze(0)

    try:

        outputs = model(input_ids.cuda())

        answer_vectors.append(outputs[0].detach().cpu().numpy().max(axis = 1))

    except:

        answer_vectors.append(np.zeros(outputs[0].cpu().detach().numpy().max(axis = 1)).shape)

    

for answer in tqdm.tqdm(answer_ids_test):

    input_ids = torch.Tensor(answer).to(torch.int64).unsqueeze(0)

    try:

        outputs = model(input_ids.cuda())

        answer_vectors_test.append(outputs[0].detach().cpu().numpy().max(axis = 1))

    except:

        answer_vectors_test.append(np.zeros(outputs[0].detach().cpu().numpy().max(axis = 1)).shape)
tfidf = TfidfVectorizer(ngram_range=(1, 3))

tsvd = TruncatedSVD(n_components = 50)

question_title = tfidf.fit_transform(train["question_title"].values)

question_title_test = tfidf.transform(test["question_title"].values)

question_title = tsvd.fit_transform(question_title)

question_title_test = tsvd.transform(question_title_test)



question_body = tfidf.fit_transform(train["question_body"].values)

question_body_test = tfidf.transform(test["question_body"].values)

question_body = tsvd.fit_transform(question_body)

question_body_test = tsvd.transform(question_body_test)



answer = tfidf.fit_transform(train["answer"].values)

answer_test = tfidf.transform(test["answer"].values)

answer = tsvd.fit_transform(answer)

answer_test = tsvd.transform(answer_test)
# from gensim import utils

from tqdm import tqdm

# from gensim.models.keyedvectors import KeyedVectors
# import gc

# def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



# def load_news(embed_dir = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'):

#     embeddings_index = KeyedVectors.load_word2vec_format(embed_dir, binary=True)

#     emb_ind = {}

#     for i, vec in tqdm(enumerate(embeddings_index.wv.vectors)):

#         emb_ind[embeddings_index.wv.index2word[i]] = vec

#     del embeddings_index

#     gc.collect()

#     return emb_ind
# vector_lookup = load_news()
# null_vector = vector_lookup["the"]

# def make_bov(sentence):

#     sent_vec = np.zeros((300))

#     sentence = sentence.split()

#     sent_length = len(sentence)

#     if sent_length == 0:

#         sent_length = 1

#     for word in sentence:

#         try:

#             sent_vec += vector_lookup[word.lower()]

#         except:

#             sent_vec += null_vector

#     sent_vec /= sent_length

#     return sent_vec
# question_title_bov = np.array([make_bov(sent) for sent in train["question_title"].values])

# question_title_bov_test = np.array([make_bov(sent) for sent in test["question_title"].values])



# question_bov = np.array([make_bov(sent) for sent in train["question_body"].values])

# question_bov_test = np.array([make_bov(sent) for sent in test["question_body"].values])



# answer_bov = np.array([make_bov(sent) for sent in train["answer"].values])

# answer_bov_test = np.array([make_bov(sent) for sent in test["answer"].values])
# question_title_len = np.array([len(sent.split()) + 1 for sent in train["question_title"].values])[:, None]

# question_title_len_test = np.array([len(sent.split()) + 1 for sent in test["question_title"].values])[:, None]



# question_len = np.array([len(sent.split()) + 1 for sent in train["question_body"].values])[:, None]

# question_len_test = np.array([len(sent.split()) + 1 for sent in test["question_body"].values])[:, None]



# answer_len = np.array([len(sent.split()) + 1 for sent in train["answer"].values])[:, None]

# answer_len_test = np.array([len(sent.split()) + 1 for sent in test["answer"].values])[:, None]
# category_means_map = train.groupby("category")[target_cols].mean().T.to_dict()

# category_te = train["category"].map(category_means_map).apply(pd.Series)

# category_te_test = test["category"].map(category_means_map).apply(pd.Series)
# train_features = np.concatenate([question_title, question_body, answer#, category_te.values

#                                 ], axis = 1)

# test_features = np.concatenate([question_title_test, question_body_test, answer_test#, category_te_test.values

#                                ], axis = 1)



# train_features = np.concatenate([question_title, question_body, answer,

#                                 question_title_bov, question_bov, answer_bov,

#                                  question_title_len, question_len, answer_len,

#                                  #, category_te.values

#                                 ], axis = 1)

# test_features = np.concatenate([question_title_test, question_body_test, answer_test,

#                                 question_title_bov_test, question_bov_test, answer_bov_test,

#                                 question_title_len_test, question_len_test, answer_len_test,

#                                 #, category_te_test.values

#                                ], axis = 1)



# train_features = np.array(question_body_vectors)[:, 0, :]

# test_features = np.array(question_body_vectors_test)[:, 0, :]





train_features = np.concatenate([question_title, question_body, answer,

                                 np.array(question_body_vectors)[:, 0, :],

                                np.array(question_title_vectors)[:, 0, :],

                                np.array(answer_vectors)[:, 0, :]

                               ], axis = 1)

test_features = np.concatenate([question_title_test, question_body_test, answer_test,

                                np.array(question_body_vectors_test)[:, 0, :],

                                np.array(question_title_vectors_test)[:, 0, :],

                                np.array(answer_vectors_test)[:, 0, :]

                               ], axis = 1)
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from sklearn.model_selection import KFold

from keras.callbacks.callbacks import EarlyStopping

from scipy.stats import spearmanr



num_folds = 5

fold_scores = []

kf = KFold(n_splits = num_folds, shuffle = True, random_state = 42)

test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):

    train_X = train_features[train_index, :]

    train_y = train[target_cols].iloc[train_index]

    

    val_X = train_features[val_index, :]

    val_y = train[target_cols].iloc[val_index]

    

    model = Sequential([

        Dense(2048, input_shape=(train_features.shape[1],)),

        Activation('relu'),

        Dense(1024),

        Activation('relu'),

        Dense(len(target_cols)),

        Activation('sigmoid'),

    ])

    

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    model.compile(optimizer='adam',

                  loss='binary_crossentropy')

    

    model.fit(train_X, train_y, epochs = 50, validation_data=(val_X, val_y), callbacks = [es])

    preds = model.predict(val_X)

    overall_score = 0

    for col_index, col in enumerate(target_cols):

        overall_score += spearmanr(preds[:, col_index], val_y[col].values).correlation/len(target_cols)

        print(col, spearmanr(preds[:, col_index], val_y[col].values).correlation)

    fold_scores.append(overall_score)

    print(overall_score)



    test_preds += model.predict(test_features)/num_folds

    

print(fold_scores)
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
sub.shape
for col_index, col in enumerate(target_cols):

    sub[col] = test_preds[:, col_index]
sub.to_csv("submission.csv", index = False)