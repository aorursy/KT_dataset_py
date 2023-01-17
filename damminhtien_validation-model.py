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
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions

# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
# prepare tokenizer on train set

# load training dataset (6K)
filename = '../input/8kflickrfeature/flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('../input/text-data-exploxe/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare val set

# load val set
filename = '../input/8kflickrfeature/flickr8k_text/Flickr_8k.devImages.txt'
val = load_set(filename)
print('Dataset: %d' % len(val))
# descriptions
val_descriptions = load_clean_descriptions('../input/text-data-exploxe/descriptions.txt', val)
print('Descriptions: val=%d' % len(val_descriptions))

# load the model 
# VGG16
val_features = load_photo_features('../input/flirck-8k-dataset-explore-image/features.pkl', val)
print('Photos: val=%d' % len(val_features))
for i in range(17,20):
    # photo features
    print('\nEncoder VGG16 %d epochs: ' % i);
    filename = '../input/development-model/model_' + str(i) + '.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, val_descriptions, val_features, tokenizer, max_length)
# load the model 
# Resnet50
val_features = load_photo_features('../input/flirck-8k-dataset-explore-image-resnet/features.pkl', val)
print('Photos: val=%d' % len(val_features))
for i in range(17,20):
    # photo features
    print('\nEncoder Resnet50 %d epochs: ' % i);
    filename = '../input/development-model-resnet50/model_' + str(i) + '.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, val_descriptions, val_features, tokenizer, max_length)
# load the model 
# Densenet121
val_features = load_photo_features('../input/flirck-8k-dataset-explore-image-desnet121/features.pkl', val)
print('Photos: val=%d' % len(val_features))
for i in range(17,20):
    # photo features
    print('\nEncoder Densenet121 %d epochs: ' % i);
    filename = '../input/development-model-densenet121/model_' + str(i) + '.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, val_descriptions, val_features, tokenizer, max_length)
# load the model 
# Inceptionv3
val_features = load_photo_features('../input/flirck-8k-dataset-explore-image-inceptionv3/features.pkl', val)
print('Photos: val=%d' % len(val_features))
for i in range(17,20):
    # photo features
    print('\nEncoder Inceptionv3 %d epochs: ' % i);
    filename = '../input/development-model-inceptionv3/model_' + str(i) + '.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, val_descriptions, val_features, tokenizer, max_length)