import os
import time
import numpy as np 
import pandas as pd 

import math
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_df = pd.read_csv("../input/train.csv")

print("Train shape : ",train_df.shape)

X_train = train_df["question_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=100_000)
tokenizer.fit_on_texts(list(X_train))
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]
nb_words = 100_000
embedding_matrix_glove = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in tokenizer.word_index.items():
    if i >= 100_000:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix_glove[i] = embedding_vector
# adapted from https://stackoverflow.com/questions/37558899/efficiently-finding-closest-word-in-tensorflow-embedding
import tensorflow as tf

batch_size = 10_000
n_neighbors = 10
closest_words = np.zeros((nb_words, n_neighbors+1))

embedding = tf.placeholder(tf.float32, [nb_words, embed_size])
batch_array = tf.placeholder(tf.float32, [batch_size, embed_size])
normed_embedding = tf.nn.l2_normalize(embedding, dim=1)
normed_array = tf.nn.l2_normalize(batch_array, dim=1)
cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embedding))
closest_k_words = tf.nn.top_k(cosine_similarity,k=n_neighbors+1)

with tf.Session() as session:
    start_idx = 0
    for end_idx in range(batch_size, nb_words, batch_size):
        print(end_idx)
        result = session.run(closest_k_words, feed_dict={embedding: embedding_matrix_glove, batch_array: embedding_matrix_glove[start_idx:end_idx]})
        closest_words[start_idx:end_idx] = result[1]

        start_idx = end_idx
index_to_word = {v:k for k,v in tokenizer.word_index.items()}
index_to_word[0] = "<PAD>"
synonyms = {index_to_word[int(x[0])]: [index_to_word[int(y)] for y in x[1:]] for x in closest_words}
synonyms["king"]
synonyms["quora"]
