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
import re

from sklearn.utils import shuffle

import numpy as np

import bz2

from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras.layers import *

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
#function to load labels and content

def splitReviewsLabels(lines):

    reviews = []

    labels = []

    for review in tqdm(lines):

        rev = reviewToX(review)

        label = reviewToY(review)

        reviews.append(rev[:512]) #取前512个单词？

        labels.append(label)

    return reviews, labels



def reviewToY(review):

    return [1,0] if review.split(' ')[0] == '__label__1' else [0,1] 



def reviewToX(review):

    review = review.split(' ', 1)[1][:-1].lower()

    review = re.sub('\d','0',review)

    review = re.sub("[^a-zA-Z]", " ",review) #remove non-character

    if 'www.' in review or 'http:' in review or 'https:' in review or '.com' in review:

        review = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", review)

    return review
train_file = bz2.BZ2File('../input/amazonreviews/train.ft.txt.bz2')

test_file = bz2.BZ2File('../input/amazonreviews/test.ft.txt.bz2')

train_lines = train_file.readlines()

test_lines = test_file.readlines()

train_lines = [x.decode('utf-8') for x in train_lines]

test_lines = [x.decode('utf-8') for x in test_lines]
# Load from the file

reviews_train, y_train = splitReviewsLabels(train_lines)

reviews_test, y_test = splitReviewsLabels(test_lines)

reviews_train, y_train = shuffle(reviews_train, y_train)

reviews_test, y_test = shuffle(reviews_test, y_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
max_features = 8192

maxlen = 128

embed_size = 100
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(reviews_train)

token_train = tokenizer.texts_to_sequences(reviews_train)

token_test = tokenizer.texts_to_sequences(reviews_test)

x_train = pad_sequences(token_train, maxlen=maxlen, padding='post')

x_test = pad_sequences(token_test, maxlen=maxlen, padding='post')
EMBEDDING_FILE = '../input/glovetwitter/glove.twitter.27B.100d.txt'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

#change below line if computing normal stats is too slow

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
baseline = tf.keras.Sequential([

    Embedding(max_features, embed_size , input_shape=(maxlen,),weights=[embedding_matrix], trainable=True),

    Flatten(),

    Dense(2, activation='softmax')

])

baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

baseline_his = baseline.fit(x_train, y_train, batch_size=10000, epochs=6, validation_split=0.1,verbose=1)
new_model = baseline
baseline.save_weights('baseline_weight')
pred_prob = new_model.predict(x_test)

ground_truth = y_test[:,1]

y_pred = pred_prob[:,1] > 0.5

y_pred = y_pred * 1 #convert boolean to 

boolean_mask = (y_pred != ground_truth)

wrong_content = []

wrong_label = []

wrong_pred = []

wrong_prob = []

for i,b in enumerate(boolean_mask):

    if b == True:

        wrong_content.append(reviews_test[i])

        wrong_label.append(ground_truth[i])

        wrong_pred.append(y_pred[i])

        wrong_prob.append(pred_prob[i,1])
wrong_pred_dict = {'content':wrong_content,'label':wrong_label,'pred':wrong_pred,'prob':wrong_prob}

wrong_df = pd.DataFrame(data=wrong_pred_dict)
from sklearn.metrics import confusion_matrix

confusion_matrix(ground_truth, y_pred)