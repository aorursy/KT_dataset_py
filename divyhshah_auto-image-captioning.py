!pip install gTTS
# Load libraries

import matplotlib.pyplot as plt

import pandas as pd

import pickle

import numpy as np

import os

from keras.applications.resnet50 import ResNet50

from keras.optimizers import Adam

from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate

from keras.models import Sequential, Model

from keras.utils import np_utils

import random

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image, sequence

import matplotlib.pyplot as plt

#from gtts import gTTS

import string
from os import listdir

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.models import Model

from keras.preprocessing.text import Tokenizer

from numpy import array

from pickle import load

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

from keras.layers import Dropout

from keras.layers.merge import add

from keras.callbacks import ModelCheckpoint

from pickle import load

from pickle import dump
# Load data

images_dir = os.listdir("../input/flickr8k/Flickr_Data/Flickr_Data")



images_path = '../input/flickr_data/Flickr_Data/Images/'

captions_path = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

train_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'

val_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'

test_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'



captions = open(captions_path, 'r').read().split("\n")

x_train = open(train_path, 'r').read().split("\n")

x_val = open(val_path, 'r').read().split("\n")

x_test = open(test_path, 'r').read().split("\n")
# extract features from each photo in the directory

def extract_features(directory):

    # load the model

    model = VGG16()

	# re-structure the model

    model.layers.pop()

    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

	# summarize

    print(model.summary())

	# extract features from each photo

    features = dict()

    for name in listdir(directory):

        # load an image from file

        filename = directory + '/' + name

        image = load_img(filename, target_size=(224, 224))

        # convert the image pixels to a numpy array

        image = img_to_array(image)

        # reshape data for the model

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # prepare the image for the VGG model

        image = preprocess_input(image)

        # get features

        feature = model.predict(image, verbose=0)

        # get image id

        image_id = name.split('.')[0]

        # store feature

        features[image_id] = feature

        print('>%s' % name)

    return features



# extract features from all images

directory = '../input/flickr8k/flickr_data/Flickr_Data/Images/'

features = extract_features(directory)

print('Extracted Features: %d' % len(features))

# save to file

with open('features.pickle', 'wb') as f:

    pickle.dump(features,f)

import string



# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, 'r')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text



# extract descriptions for images

def load_descriptions(doc):

	mapping = dict()

	# process lines

	for line in doc.split('\n'):

		# split line by white space

		tokens = line.split()

		if len(line) < 2:

			continue

		# take the first token as the image id, the rest as the description

		image_id, image_desc = tokens[0], tokens[1:]

		# remove filename from image id

		image_id = image_id.split('.')[0]

		# convert description tokens back to string

		image_desc = ' '.join(image_desc)

		# create the list if needed

		if image_id not in mapping:

			mapping[image_id] = list()

		# store description

		mapping[image_id].append(image_desc)

	return mapping



def clean_descriptions(descriptions):

	# prepare translation table for removing punctuation

    

    table = str.maketrans('','',string.punctuation)

    for key,desc_list in descriptions.items():

        for i in range(len(desc_list)):

            desc = desc_list[i]

			# tokenize

            desc = desc.split()

			# convert to lower case

            desc = [word.lower() for word in desc]

			# remove punctuation from each token

            desc = [w.translate(table) for w in desc]

			# remove hanging 's' and 'a'

            desc = [word for word in desc if len(word)>1]

			# remove tokens with numbers in them

            desc = [word for word in desc if word.isalpha()]

            # store as string

            desc_list[i] =  ' '.join(desc)



# convert the loaded descriptions into a vocabulary of words

def to_vocabulary(descriptions):

	# build a list of all description strings

	all_desc = set()

	for key in descriptions.keys():

		[all_desc.update(d.split()) for d in descriptions[key]]

	return all_desc



# save descriptions to file, one per line

def save_descriptions(descriptions, filename):

	lines = list()

	for key, desc_list in descriptions.items():

		for desc in desc_list:

			lines.append(key + ' ' + desc)

	data = '\n'.join(lines)

	file = open(filename, 'w')

	file.write(data)

	file.close()



filename = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

# load descriptions

doc = load_doc(filename)

#parse descriptions

descriptions = load_descriptions(doc)

print('Loaded: %d ' % len(descriptions))

# clean descriptions

clean_descriptions(descriptions)

# summarize vocabulary

vocabulary = to_vocabulary(descriptions)

print('Vocabulary Size: %d' % len(vocabulary))

# save to file

save_descriptions(descriptions, 'descriptions.txt')
from pickle import load

 

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

 

# load training dataset (6K)

filename = train_path

train = load_set(filename)

print('Dataset: %d' % len(train))

# descriptions

train_descriptions = load_clean_descriptions('../working/descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

# photo features

train_features = load_photo_features('../working/features.pickle', train)

print('Photos: train=%d' % len(train_features))
# convert a dictionary of clean descriptions to a list of descriptions

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



# prepare tokenizer

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)
# calculate the length of the description with the most words

def max_length(descriptions):

	lines = to_lines(descriptions)

	return max(len(d.split()) for d in lines)
# define the captioning model

def define_model(vocab_size, max_length):

	# feature extractor model

	inputs1 = Input(shape=(4096,))

	fe1 = Dropout(0.5)(inputs1)

	fe2 = Dense(256, activation='relu')(fe1)

	# sequence model

	inputs2 = Input(shape=(max_length,))

	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)

	se2 = Dropout(0.5)(se1)

	se3 = LSTM(256)(se2)

	# decoder model

	decoder1 = add([fe2, se3])

	decoder2 = Dense(256, activation='relu')(decoder1)

	outputs = Dense(vocab_size, activation='softmax')(decoder2)

	# tie it together [image, seq] [word]

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)

	model.compile(loss='categorical_crossentropy', optimizer='adam')

	# summarize model

	print(model.summary())

	plot_model(model, to_file='model.png', show_shapes=True)

	return model
# data generator, intended to be used in a call to model.fit_generator()

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):

	# loop for ever over images

	while 1:

		for key, desc_list in descriptions.items():

			# retrieve the photo feature

			photo = photos[key][0]

			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)

			yield [[in_img, in_seq], out_word]
# create sequences of images, input sequences and output words for an image

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):

	X1, X2, y = list(), list(), list()

	# walk through each description for the image

	for desc in desc_list:

		# encode the sequence

		seq = tokenizer.texts_to_sequences([desc])[0]

		# split one sequence into multiple X,y pairs

		for i in range(1, len(seq)):

			# split into input and output pair

			in_seq, out_seq = seq[:i], seq[i]

			# pad input sequence

			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

			# encode output sequence

			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

			# store

			X1.append(photo)

			X2.append(in_seq)

			y.append(out_seq)

	return array(X1), array(X2), array(y)
filename = train_path

train = load_set(filename)

print('Dataset: %d' % len(train))

# descriptions

train_descriptions = load_clean_descriptions('../working/descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

# photo features

train_features = load_photo_features('../working/features.pickle', train)

print('Photos: train=%d' % len(train_features))

# prepare tokenizer

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length

max_length = max_length(train_descriptions)

print('Description Length: %d' % max_length)

# prepare sequences

X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
# train the model, run epochs manually and save after each epoch

model = define_model(vocab_size, max_length)

epochs = 20

steps = len(train_descriptions)

for i in range(epochs):

	# create the data generator

	generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)

	# fit for one epoch

	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

	# save model

	model.save('model_' + str(i) + '.h5')
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

filename = train_path

train = load_set(filename)

print('Dataset: %d' % len(train))

# descriptions

train_descriptions = load_clean_descriptions('../working/descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length

max_length = max_length(train_descriptions)

print('Description Length: %d' % max_length)



# prepare test set



# load test set

filename = test_path

test = load_set(filename)

print('Dataset: %d' % len(test))

# descriptions

test_descriptions = load_clean_descriptions('../working/descriptions.txt', test)

print('Descriptions: test=%d' % len(test_descriptions))

# photo features

test_features = load_photo_features('../working/features.pickle', test)

print('Photos: test=%d' % len(test_features))



# load the modelfilename = 'model-ep002-loss3.245-val_loss3.612.h5'

for i in range(0,20):

    filename=f'model_{i}.h5'

    #filename='model_2.h5'

    model = load_model(filename)

    # evaluate model

    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
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
# load training dataset (6K)

filename = captions_path

train = load_set(filename)

print('Dataset: %d' % len(train))

# descriptions

train_descriptions = load_clean_descriptions('../working/descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer

tokenizer = create_tokenizer(train_descriptions)

# save the tokenizer

dump(tokenizer, open('tokenizer.pkl', 'wb'))
from pickle import load

from numpy import argmax

from keras.preprocessing.sequence import pad_sequences

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.models import Model

from keras.models import load_model



# extract features from each photo in the directory

def extract_features(filename):

	# load the model

	model = VGG16()

	# re-structure the model

	model.layers.pop()

	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

	# load the photo

	image = load_img(filename, target_size=(224, 224))

	# convert the image pixels to a numpy array

	image = img_to_array(image)

	# reshape data for the model

	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

	# prepare the image for the VGG model

	image = preprocess_input(image)

	# get features

	feature = model.predict(image, verbose=0)

	return feature



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



# load the tokenizer

tokenizer = load(open('../working/tokenizer.pkl', 'rb'))

# pre-define the max sequence length (from training)

max_length = 34

# load the model

model = load_model('../working/model_3.h5')

# load and prepare the photograph

photo = extract_features('../input/test-images-caption/t3.jpg')

# generate description

description = generate_desc(model, tokenizer, photo, max_length)

print(description)
# Loading captions as values and images as key in dictionary

tokens = {}



for ix in range(len(captions)-1):

    temp = captions[ix].split("#")

    if temp[0] in tokens:

        tokens[temp[0]].append(temp[1][2:])

    else:

        tokens[temp[0]] = [temp[1][2:]]
# displaying an image and captions given to it

temp = captions[78].split("#")

from IPython.display import Image, display

z = Image(filename=images_path+temp[0])

display(z)



for ix in range(len(tokens[temp[0]])):

    print(tokens[temp[0]][ix])
# Creating train, test and validation dataset files with header as 'image_id' and 'captions'

train_dataset = open('flickr_8k_train_dataset.txt','wb')

train_dataset.write(b"image_id\tcaptions\n")



val_dataset = open('flickr_8k_val_dataset.txt','wb')

val_dataset.write(b"image_id\tcaptions\n")



test_dataset = open('flickr_8k_test_dataset.txt','wb')

test_dataset.write(b"image_id\tcaptions\n")
# Populating the above created files for train, test and validation dataset with image ids and captions for each of these images

for img in x_train:

    if img == '':

        continue

    for capt in tokens[img]:

        caption = "<start> "+ capt + " <end>"

        train_dataset.write((img+"\t"+caption+"\n").encode())

        train_dataset.flush()

train_dataset.close()



for img in x_test:

    if img == '':

        continue

    for capt in tokens[img]:

        caption = "<start> "+ capt + " <end>"

        test_dataset.write((img+"\t"+caption+"\n").encode())

        test_dataset.flush()

test_dataset.close()



for img in x_val:

    if img == '':

        continue

    for capt in tokens[img]:

        caption = "<start> "+ capt + " <end>"

        val_dataset.write((img+"\t"+caption+"\n").encode())

        val_dataset.flush()

val_dataset.close()
# Loading 50 layer Residual Network Model and getting the summary of the model

from IPython.core.display import display, HTML

display(HTML("""<a href="http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006">ResNet50 Architecture</a>"""))

model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

model.summary()

# Note: For more details on ResNet50 architecture you can click on hyperlink given below
# Helper function to process images

def preprocessing(img_path):

    im = image.load_img(img_path, target_size=(224,224,3))

    im = image.img_to_array(im)

    im = np.expand_dims(im, axis=0)

    return im
train_data = {}

ctr=0

for ix in x_train:

    if ix == "":

        continue

    if ctr >= 3000:

        break

    ctr+=1

    if ctr%1000==0:

        print(ctr)

    path = images_path + ix

    img = preprocessing(path)

    pred = model.predict(img).reshape(2048)

    train_data[ix] = pred
train_data['2513260012_03d33305cf.jpg'].shape
# opening train_encoded_images.p file and dumping it's content

with open( "train_encoded_images.p", "wb" ) as pickle_f:

    pickle.dump(train_data, pickle_f )  
# Loading image and its corresponding caption into a dataframe and then storing values from dataframe into 'ds'

pd_dataset = pd.read_csv("flickr_8k_train_dataset.txt", delimiter='\t')

ds = pd_dataset.values

print(ds.shape)
pd_dataset.head()
# Storing all the captions from ds into a list

sentences = []

for ix in range(ds.shape[0]):

    sentences.append(ds[ix, 1])

    

print(len(sentences))
# First 5 captions stored in sentences

sentences[:5]
# Splitting each captions stored in 'sentences' and storing them in 'words' as list of list

words = [i.split() for i in sentences]
# Creating a list of all unique words

unique = []

for i in words:

    unique.extend(i)

unique = list(set(unique))



print(len(unique))



vocab_size = len(unique)
# Vectorization

word_2_indices = {val:index for index, val in enumerate(unique)}

indices_2_word = {index:val for index, val in enumerate(unique)}
word_2_indices['UNK'] = 0

word_2_indices['raining'] = 8253
indices_2_word[0] = 'UNK'

indices_2_word[8253] = 'raining'
print(word_2_indices['<start>'])

print(indices_2_word[4011])

print(word_2_indices['<end>'])

print(indices_2_word[8051])
vocab_size = len(word_2_indices.keys())

print(vocab_size)
max_len = 0



for i in sentences:

    i = i.split()

    if len(i) > max_len:

        max_len = len(i)



print(max_len)
padded_sequences, subsequent_words = [], []



for ix in range(ds.shape[0]):

    partial_seqs = []

    next_words = []

    text = ds[ix, 1].split()

    text = [word_2_indices[i] for i in text]

    for i in range(1, len(text)):

        partial_seqs.append(text[:i])

        next_words.append(text[i])

    padded_partial_seqs = sequence.pad_sequences(partial_seqs, max_len, padding='post')



    next_words_1hot = np.zeros([len(next_words), vocab_size], dtype=np.bool)

    

    #Vectorization

    for i,next_word in enumerate(next_words):

        next_words_1hot[i, next_word] = 1

        

    padded_sequences.append(padded_partial_seqs)

    subsequent_words.append(next_words_1hot)

    

padded_sequences = np.asarray(padded_sequences)

subsequent_words = np.asarray(subsequent_words)



print(padded_sequences.shape)

print(subsequent_words.shape)
print(padded_sequences[0])
for ix in range(len(padded_sequences[0])):

    for iy in range(max_len):

        print(indices_2_word[padded_sequences[0][ix][iy]],)

    print("\n")



print(len(padded_sequences[0]))
num_of_images = 2000
captions = np.zeros([0, max_len])

next_words = np.zeros([0, vocab_size])
for ix in range(num_of_images):#img_to_padded_seqs.shape[0]):

    captions = np.concatenate([captions, padded_sequences[ix]])

    next_words = np.concatenate([next_words, subsequent_words[ix]])



np.save("captions.npy", captions)

np.save("next_words.npy", next_words)



print(captions.shape)

print(next_words.shape)
with open('../input/train_encoded_images.p', 'rb') as f:

    encoded_images = pickle.load(f, encoding="bytes")
imgs = []



for ix in range(ds.shape[0]):

    if ds[ix, 0].encode() in encoded_images.keys():

        

        print(ix, encoded_images[ds[ix, 0].encode()])

        imgs.append(list(encoded_images[ds[ix, 0].encode()]))



imgs = np.asarray(imgs)

print(imgs.shape)
images = []



for ix in range(num_of_images):

    for iy in range(padded_sequences[ix].shape[0]):

        images.append(imgs[ix])

        

images = np.asarray(images)



np.save("images.npy", images)



print(images.shape)
image_names = []



for ix in range(num_of_images):

    for iy in range(padded_sequences[ix].shape[0]):

        image_names.append(ds[ix, 0])

        

image_names = np.asarray(image_names)



np.save("image_names.npy", image_names)



print(len(image_names))
#here we loaded the previously stored numpy array which contain the captions 

captions = np.load("captions.npy")

next_words = np.load("next_words.npy")



print(captions.shape)

print(next_words.shape)
images = np.load("images.npy")



print(images.shape)
imag = np.load("image_names.npy")

        

print(imag.shape)
embedding_size = 128

max_len = 40
image_model = Sequential()



image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))

image_model.add(RepeatVector(max_len))



image_model.summary()
language_model = Sequential()



language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))

language_model.add(LSTM(256, return_sequences=True))

language_model.add(TimeDistributed(Dense(embedding_size)))



language_model.summary()
conca = Concatenate()([image_model.output, language_model.output])

x = LSTM(128, return_sequences=True)(conca)

x = LSTM(512, return_sequences=False)(x)

x = Dense(vocab_size)(x)

out = Activation('softmax')(x)

model = Model(inputs=[image_model.input, language_model.input], outputs = out)



# model.load_weights("../input/model_weights.h5")

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.summary()
hist = model.fit([images, captions], next_words, batch_size=512, epochs=200)
import json



model_json = model.to_json()

with open("model_in_json.json", "w") as json_file:

    json.dump(model_json, json_file)
model.save_weights("model_weights.h5")
def preprocessing(img_path):

    im = image.load_img(img_path, target_size=(224,224,3))

    im = image.img_to_array(im)

    im = np.expand_dims(im, axis=0)

    return im
def get_encoding(model, img):

    image = preprocessing(img)

    pred = model.predict(image).reshape(2048)

    return pred
resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
img = "../input/Flickr_Data/Flickr_Data/Images/1045521051_108ebc19be.jpg"



test_img = get_encoding(resnet, img)
def predict_captions(image):

    start_word = ["<start>"]

    while True:

        par_caps = [word_2_indices[i] for i in start_word]

        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')

        preds = model.predict([np.array([image]), np.array(par_caps)])

        word_pred = indices_2_word[np.argmax(preds[0])]

        start_word.append(word_pred)

        

        if word_pred == "<end>" or len(start_word) > max_len:

            break

            

    return ' '.join(start_word[1:-1])



Argmax_Search = predict_captions(test_img)
import os

import IPython

language='en-in'

mytext=Argmax_Search

myobj = gTTS(text=mytext, lang=language, slow=False) 

myobj.save("caption.mp3") 

z = Image(filename=img)

display(z)



#print("Predicted caption is ="+Argmax_Search)

IPython.display.display(IPython.display.Audio('caption.mp3'))

print(mytext)