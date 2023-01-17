!pip install bert-for-tf2

!pip install sentencepiece
import re

import os

import bert

import math

import random

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from matplotlib import pyplot as plt
movie_reviews = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

movie_reviews.isnull().values.any()

movie_reviews.shape
def preprocess_text(sen):

    # Removing html tags

    sentence = remove_tags(sen)

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
TAG_RE = re.compile(r'<[^>]+>')



def remove_tags(text):

    return TAG_RE.sub('', text)
reviews = []

sentences = list(movie_reviews['review'])

for sen in sentences:

    reviews.append(preprocess_text(sen))
print(movie_reviews.columns.values)
movie_reviews.sentiment.unique()
y = movie_reviews['sentiment']



y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
print(reviews[10])
print(y[10])
BertTokenizer = bert.bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer(

    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",

    trainable=False

)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
tokenizer.tokenize("don't be so judgmental")
tokenizer.convert_tokens_to_ids(tokenizer.tokenize("dont be so judgmental"))
def tokenize_reviews(text_reviews):

    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))
tokenized_reviews = [tokenize_reviews(review) for review in reviews]
reviews_with_len = [[review, y[i], len(review)] for i, review in enumerate(tokenized_reviews)]
random.shuffle(reviews_with_len)
reviews_with_len.sort(key=lambda x: x[2])
sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))

processed_dataset
BATCH_SIZE = 32

batched_dataset = processed_dataset.padded_batch(

    BATCH_SIZE, padded_shapes=((None, ), ())

)
TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)

TEST_BATCHES = TOTAL_BATCHES // 10

batched_dataset.shuffle(TOTAL_BATCHES)

test_data = batched_dataset.take(TEST_BATCHES)

train_data = batched_dataset.skip(TEST_BATCHES)
class TEXT_MODEL(tf.keras.Model):

    

    def __init__(self,

                 vocabulary_size,

                 embedding_dimensions=128,

                 cnn_filters=50,

                 dnn_units=512,

                 model_output_classes=2,

                 dropout_rate=0.1,

                 training=False,

                 name="text_model"):

        super(TEXT_MODEL, self).__init__(name=name)

        

        self.embedding = tf.keras.layers.Embedding(vocabulary_size,

                                          embedding_dimensions)

        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=cnn_filters,

                                        kernel_size=2,

                                        padding="valid",

                                        activation="relu")

        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=cnn_filters,

                                        kernel_size=3,

                                        padding="valid",

                                        activation="relu")

        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=cnn_filters,

                                        kernel_size=4,

                                        padding="valid",

                                        activation="relu")

        self.pool = tf.keras.layers.GlobalMaxPool1D()

        

        self.dense_1 = tf.keras.layers.Dense(units=dnn_units, activation="relu")

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        if model_output_classes == 2:

            self.last_dense = tf.keras.layers.Dense(units=1,

                                           activation="sigmoid")

        else:

            self.last_dense = tf.keras.layers.Dense(units=model_output_classes,

                                           activation="softmax")

    

    def call(self, inputs, training):

        l = self.embedding(inputs)

        l_1 = self.cnn_layer1(l) 

        l_1 = self.pool(l_1) 

        l_2 = self.cnn_layer2(l) 

        l_2 = self.pool(l_2)

        l_3 = self.cnn_layer3(l)

        l_3 = self.pool(l_3) 

        

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)

        concatenated = self.dense_1(concatenated)

        concatenated = self.dropout(concatenated, training)

        model_output = self.last_dense(concatenated)

        

        return model_output
VOCAB_LENGTH = len(tokenizer.vocab)

EMB_DIM = 200

CNN_FILTERS = 100

DNN_UNITS = 256

OUTPUT_CLASSES = 2



DROPOUT_RATE = 0.2



NB_EPOCHS = 5
text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,

                        embedding_dimensions=EMB_DIM,

                        cnn_filters=CNN_FILTERS,

                        dnn_units=DNN_UNITS,

                        model_output_classes=OUTPUT_CLASSES,

                        dropout_rate=DROPOUT_RATE)
if OUTPUT_CLASSES == 2:

    text_model.compile(loss="binary_crossentropy",

                       optimizer="adam",

                       metrics=["accuracy"])

else:

    text_model.compile(loss="sparse_categorical_crossentropy",

                       optimizer="adam",

                       metrics=["sparse_categorical_accuracy"])
text_model.fit(train_data, epochs=NB_EPOCHS)
results = text_model.evaluate(test_data)

print(results)