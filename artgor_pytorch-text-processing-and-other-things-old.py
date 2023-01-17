import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import numpy as np

import pandas as pd

import os

import time

from tqdm import tqdm_notebook

import pickle

import gc

from sklearn.model_selection import KFold



from jigsaw_utils import *
# setting parameters.

set_seed(42)



crawl_embedding_path = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

glove_embedding_path = '../input/glove840b300dtxt/glove.840B.300d.txt'

num_models = 2

lstm_units = 128

dense_hidden_units = 4 * lstm_units

max_len = 220

embed_size = 300

max_features = 120000
load = True

if not load:

    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

    train = df_parallelize_run(train, text_clean_wrapper)

    test = df_parallelize_run(test, text_clean_wrapper)

    train.to_csv('processed_train.csv', index=False)

    test.to_csv('processed_test.csv', index=False)

else:

    train = pd.read_csv('../input/jigsaw-public-files/train.csv')

    test = pd.read_csv('../input/jigsaw-public-files/test.csv')

    # after processing some of the texts are emply

    train['comment_text'] = train['comment_text'].fillna('')

    test['comment_text'] = test['comment_text'].fillna('')
%%time

glove_embed = load_embed(glove_embedding_path)

oovs = vocab_check_coverage(train, glove_embed)
print(oovs[0]['oov_words'][:20])
if not load:

    tokenizer = text.Tokenizer(lower=False, num_words=max_features)

    tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text']))

    

    # by default tokenizer keeps all words, I leave only top max_features

    sorted_by_word_count = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)

    tokenizer.word_index = {}

    i = 0

    for word,count in sorted_by_word_count:

        if i == max_features:

            break

        tokenizer.word_index[word] = i + 1    # <= because tokenizer is 1 indexed

        i += 1

    

    with open(f'tokenizer_{max_features}.pickle', 'wb') as f:

        pickle.dump(tokenizer, f)

else:

    with open(f'../input/jigsaw-public-files/tokenizer_{max_features}.pickle', 'rb') as f:

        tokenizer = pickle.load(f)

    

X_train = tokenizer.texts_to_sequences(train['comment_text'])

X_test = tokenizer.texts_to_sequences(test['comment_text'])

x_train_lens = [len(i) for i in X_train]

x_test_lens  = [len(i) for i in X_test]
y_train = np.where(train['target'] >= 0.5, 1, 0)

y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

final_y_train = np.hstack([y_train[:, np.newaxis], y_aux_train])
crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, crawl_embedding_path, embed_size)

print('n unknown words (crawl): ', len(unknown_words_crawl))



glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, glove_embedding_path, embed_size)

print('n unknown words (glove): ', len(unknown_words_glove))



embedding_matrix = crawl_matrix * 0.5 +  glove_matrix * 0.5



del crawl_matrix

del glove_matrix

gc.collect()
embedding_matrix_small = np.zeros((embedding_matrix.shape[0], 30))
# splits for training

splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train, final_y_train))
test_preds = train_on_folds(X_train, x_train_lens, final_y_train, X_test, x_test_lens,

                            splits, embedding_matrix, embedding_matrix_small, n_epochs=2, validate=False, debug=True)
submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': test_preds.mean(1)

})



submission.to_csv('submission.csv', index=False)