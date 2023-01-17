# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Model

from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation

from keras.layers import concatenate, BatchNormalization, Input

from keras.layers.merge import add

from keras.utils import to_categorical

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.utils import plot_model



import matplotlib.pyplot as plt

import cv2

import string

import time

print("Running.....")
token_path = '/kaggle/input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

text = open(token_path, 'r', encoding = 'utf-8').read()

print(text[:500])
def load_description(text):

    mapping = dict()

    for line in text.split("\n"):

        token = line.split("\t")

        if len(line) < 2:

            continue

        img_id = token[0].split('.')[0]

        img_des = token[1]

        if img_id not in mapping:

            mapping[img_id] = list()

        mapping[img_id].append(img_des)

    return mapping



descriptions = load_description(text)

print("Number of items: " + str(len(descriptions)))
descriptions['1000268201_693b08cb0e']
def clean_description(desc):

    for key, des_list in desc.items():

        for i in range(len(des_list)):

            caption = des_list[i]

            caption = [ch for ch in caption if ch not in string.punctuation]

            caption = ''.join(caption)

            caption = caption.split(' ')

            caption = [word.lower() for word in caption if len(word)>1 and word.isalpha()]

            caption = ' '.join(caption)

            des_list[i] = caption



clean_description(descriptions)

descriptions['1000268201_693b08cb0e']
def to_vocab(desc):

    words = set()

    for key in desc.keys():

        for line in desc[key]:

            words.update(line.split())

    return words

vocab = to_vocab(descriptions)

len(vocab)
import glob

images = '/kaggle/input/flickr8k/flickr_data/Flickr_Data/Images/'

# Create a list of all image names in the directory

img = glob.glob(images + '*.jpg')

len(img)
train_path = '/kaggle/input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'

train_images = open(train_path, 'r', encoding = 'utf-8').read().split("\n")

train_img = []



for im in img:

    if(im[len(images):] in train_images):

        train_img.append(im)
test_path = '/kaggle/input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'

test_images = open(test_path, 'r', encoding = 'utf-8').read().split("\n")

test_img = []



for im in img:

    if(im[len(images): ] in test_images):

        test_img.append(im)

len(test_img)
#load descriptions of train and test set separately

def load_clean_descriptions(des, dataset):

    dataset_des = dict()

    for key, des_list in des.items():

        if key+'.jpg' in dataset:

            if key not in dataset_des:

                dataset_des[key] = list()

            for line in des_list:

                desc = 'startseq ' + line + ' endseq'

                dataset_des[key].append(desc)

    return dataset_des



train_descriptions = load_clean_descriptions(descriptions, train_images)

print('Descriptions: train=%d' % len(train_descriptions))
train_descriptions['1000268201_693b08cb0e']
from keras.preprocessing.image import load_img, img_to_array

def preprocess_img(img_path):

    #inception v3 excepts img in 299*299

    img = load_img(img_path, target_size = (299, 299))

    x = img_to_array(img)

    # Add one more dimension

    x = np.expand_dims(x, axis = 0)

    x = preprocess_input(x)

    return x
base_model = InceptionV3(weights = 'imagenet')

base_model.summary()
model = Model(base_model.input, base_model.layers[-2].output)
#function to encode an image into a vector using inception v3

def encode(image):

    image = preprocess_img(image)

    vec = model.predict(image)

    vec = np.reshape(vec, (vec.shape[1]))

    return vec
#run the encode function on all train images

start = time.time()

encoding_train = {}

for img in train_img:

    encoding_train[img[len(images):]] = encode(img)

print("Time Taken is: " + str(time.time() - start))
#Encode all the test images

start = time.time()

encoding_test = {}

for img in test_img:

    encoding_test[img[len(images):]] = encode(img)

print("Time taken is: " + str(time.time() - start))
train_features = encoding_train

test_features = encoding_test

print("Train image encodings: " + str(len(train_features)))

print("Test image encodings: " + str(len(test_features)))
train_features['1000268201_693b08cb0e.jpg'].shape
#list of all training captions

all_train_captions = []

for key, val in train_descriptions.items():

    for caption in val:

        all_train_captions.append(caption)

len(all_train_captions)
#onsider only words which occur atleast 10 times

vocabulary = vocab

threshold = 10

word_counts = {}

for cap in all_train_captions:

    for word in cap.split(' '):

        word_counts[word] = word_counts.get(word, 0) + 1



vocab = [word for word in word_counts if word_counts[word] >= threshold]

print("Unique words: " + str(len(word_counts)))

print("our Vocabulary: " + str(len(vocab)))
#word mapping to integers

ixtoword = {}

wordtoix = {}



ix = 1

for word in vocab:

    wordtoix[word] = ix

    ixtoword[ix] = word

    ix += 1
vocab_size = len(ixtoword) + 1  #1 for appended zeros

vocab_size
#find the maximum length of a description in a dataset

max_length = max(len(des.split()) for des in all_train_captions)

max_length
#since there are almost 30000 descriptions to process we will use datagenerator

X1, X2, y = list(), list(), list()

for key, des_list in train_descriptions.items():

    pic = train_features[key + '.jpg']

    for cap in des_list:

        seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]

        for i in range(1, len(seq)):

            in_seq, out_seq = seq[:i], seq[i]

            in_seq = pad_sequences([in_seq], maxlen = max_length)[0]

            out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]

            #store

            X1.append(pic)

            X2.append(in_seq)

            y.append(out_seq)



X2 = np.array(X2)

X1 = np.array(X1)

y = np.array(y)

print(X1.shape)
#load glove vectors for embedding layer

embeddings_index = {}

glove = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt', 'r', encoding = 'utf-8').read()

for line in glove.split("\n"):

    values = line.split(" ")

    word = values[0]

    indices = np.asarray(values[1: ], dtype = 'float32')

    embeddings_index[word] = indices

print('Total word vectors: ' + str(len(embeddings_index)))
emb_dim = 200

emb_matrix = np.zeros((vocab_size, emb_dim))

for word, i in wordtoix.items():

    emb_vec = embeddings_index.get(word)

    if emb_vec is not None:

        emb_matrix[i] = emb_vec

emb_matrix.shape
# define the model

ip1 = Input(shape = (2048, ))

fe1 = Dropout(0.2)(ip1)

fe2 = Dense(256, activation = 'relu')(fe1)

ip2 = Input(shape = (max_length, ))

se1 = Embedding(vocab_size, emb_dim, mask_zero = True)(ip2)

se2 = Dropout(0.2)(se1)

se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])

decoder2 = Dense(256, activation = 'relu')(decoder1)

outputs = Dense(vocab_size, activation = 'softmax')(decoder2)

model = Model(inputs = [ip1, ip2], outputs = outputs)

model.summary()
model.layers[2]
model.layers[2].set_weights([emb_matrix])

model.layers[2].trainable = False

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

plot_model(model, to_file = 'model.png', show_shapes = True, show_layer_names = True)
for i in range(30):

    model.fit([X1, X2], y, epochs = 1, batch_size = 256)

    if(i%2 == 0):

        model.save_weights("image-caption-weights" + str(i) + ".h5")
def greedy_search(pic):

    start = 'startseq'

    for i in range(max_length):

        seq = [wordtoix[word] for word in start.split() if word in wordtoix]

        seq = pad_sequences([seq], maxlen = max_length)

        yhat = model.predict([pic, seq])

        yhat = np.argmax(yhat)

        word = ixtoword[yhat]

        start += ' ' + word

        if word == 'endseq':

            break

    final = start.split()

    final = final[1:-1]

    final = ' '.join(final)

    return final
pic = list(encoding_test.keys())[250]

img = encoding_test[pic].reshape(1, 2048)

x = plt.imread(images + pic)

plt.imshow(x)

plt.show()

print(greedy_search(img))
pic = list(encoding_test.keys())[570]

img = encoding_test[pic].reshape(1, 2048)

x = plt.imread(images + pic)

plt.imshow(x)

plt.show()

print(greedy_search(img))
model.save("my_model.h5")
#train it for some more time

model.fit([X1, X2], y, epochs = 1, batch_size = 64)

model.save("my_model_"+str(i)+".h5")
pic = list(encoding_test.keys())[888]

img = encoding_test[pic].reshape(1, 2048)

x = plt.imread(images + pic)

plt.imshow(x)

plt.show()

print(greedy_search(img))
model.save("my-cap.h5")