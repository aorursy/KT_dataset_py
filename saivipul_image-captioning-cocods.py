# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#IMPORTS
import gc
import string
from collections import Counter
import datetime as dt
from numpy import array
import pickle
import os
from pickle import load, dump
from keras.applications import InceptionResNetV2
from keras.models import Model, Sequential
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras import Input
from keras.layers import Dropout, Dense, Embedding, LSTM, GaussianDropout, Flatten, Convolution2D, TimeDistributed, Bidirectional, Activation, RepeatVector, Concatenate
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils, plot_model
from keras.optimizers import Adam
import random
import keras.backend as K
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import time
import json
from glob import glob
from PIL import Image
#IMPORTING THE DATASET

annotations_path="../input/coco2014/captions/annotations/"
train_annotations_raw_path=annotations_path+"captions_train2014.json"

with open(train_annotations_raw_path) as f:
    train_annotations_raw=json.load(f)
train_annotations=train_annotations_raw['annotations']
print(len(train_annotations))
print(train_annotations[:5])
#FREEING RAM

del train_annotations_raw

gc.collect()
#PROCESSING THE DATASET

path_to_train_images="../input/coco2014/train2014/train2014/"

train_captions = []
train_images_path = []
for annotation in train_annotations:
    image_id = annotation['image_id']
    caption = '<start> ' + annotation['caption'] + ' <end>'
    image_path=path_to_train_images + 'COCO_train2014_' + '%012d.jpg' %(image_id)
    
    train_captions.append(caption)
    train_images_path.append(image_path)
train_captions, train_images_path = shuffle(train_captions, train_images_path, random_state=1)
train_examples=40000
train_captions=train_captions[:train_examples]
train_images_path=train_images_path[:train_examples]
print(len(train_captions))
print(train_captions[:5])
print(len(train_images_path))
print(train_images_path[:5])
#FREEING RAM

del train_annotations

gc.collect()
#PREPROCESSING AND TOKENIZING THE CAPTIONS

vocabulary_length=8192
tokenizer = Tokenizer(num_words=vocabulary_length, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
train_sequences = tokenizer.texts_to_sequences(train_captions)
train_sequences = pad_sequences(train_sequences, padding='post')
print(len(train_sequences))

max_caption_length=0
for seq in train_sequences:
    max_caption_length = max(max_caption_length, len(seq))
print(max_caption_length)
#FREEING RAM

del tokenizer

gc.collect()
#CREATING REFINED VOCABULARY

vocabulary = {}
for i in range(train_examples):
    words = train_captions[i].split()
    for word in words:
        if(word not in vocabulary.keys()):
            vocabulary[word]=1
        else:
            vocabulary[word]+=1
refined_vocabulary = set()
for key in vocabulary.keys():
    if(vocabulary[key]>16):
        refined_vocabulary.add(key)
        
#GLOVE EMBEDDINGS

file=open("../input/glove6b200d/glove.6B.200d.txt", "r")
content=file.read()
file.close()
lines=content.split("\n")
embeddings = {}
for line in lines[:-1]:
    word_and_vector = line.split()
    word = word_and_vector[0]
    vector = np.asarray(word_and_vector[1:], dtype='float32')
    embeddings[word]=vector

embeddings_dim = 200
embeddings_matrix = np.zeros((vocabulary_length+1, embeddings_dim))
for j in range(train_examples):
    words = train_captions[j].split()
    for i in range(len(words)):
        if(words[i] in refined_vocabulary):
            vector=embeddings.get(words[i])
            if(vector is not None):
                index = train_sequences[j][i]
                embeddings_matrix[index]=vector
#FREEING RAM

del train_captions
del embeddings
del content
del vocabulary
del refined_vocabulary

gc.collect()
#LOADING INCEPTION RESNET ENCODINGS
encoded_train_images = {}
with open("../input/inceptionresnetv2-encryption/encoded_train_images.pkl", "rb") as pickle_file:
    encoded_train_images=load(pickle_file)
#DATA GENERATOR FUNCTION TO KEEP RAM FREE

def data_generator(batch_size):
    X1, X2, Y = list(), list(), list()
    n=0
    while True:
        n+=1
        for i in range(train_examples):
            feature_vec = encoded_train_images[train_images_path[i]]
            sequence = train_sequences[i]
            for i in range(1, len(sequence)):
                input_sequence = pad_sequences([sequence[:i]], maxlen=max_caption_length)[0]
                output_sequence = to_categorical([sequence[i]], num_classes=vocabulary_length+1)[0]

                X1.append(feature_vec)
                X2.append(input_sequence)
                Y.append(output_sequence)
                if(n==batch_size):
                    yield [[array(X1), array(X2)], array(Y)]
                    X1, X2, Y = list(), list(), list()
                    n=0
#CREATING IMAGE MODEL
image_model = Sequential()
image_model.add(Dense(embeddings_dim, input_shape=(1536,), activation='relu'))
image_model.add(RepeatVector(max_caption_length))
image_model.summary()
plot_model(image_model, to_file="image_model.png")
#CREATING LANGUAGE MODEL
language_model = Sequential()
language_model.add(Embedding(input_dim=vocabulary_length+1, output_dim=embeddings_dim, input_length=max_caption_length))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embeddings_dim)))
language_model.summary()
plot_model(language_model, to_file="language_model.png")
#COMBINING IMAGE MODEL AND LANGUAGE MODEL
concatenation_layer = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(concatenation_layer)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocabulary_length)(x)
output = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = output, name="model")

model.summary()
plot_model(model, to_file="model.png")
#EMBEDDINGS LAYER IS NOT TRAINED
model.layers[2].set_weights([embeddings_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
###### TRAINING SESSION
batch_size = 16
steps = train_examples//batch_size
generator = data_generator(batch_size)
hist = model.fit_generator(generator, epochs=100, steps_per_epoch=steps, verbose=1)
