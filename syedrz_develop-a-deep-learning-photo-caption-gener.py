# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from os import listdir

from pickle import dump

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.models import Model

 

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

directory = '/kaggle/input/flickr8k_Dataset/Flickr8k_Dataset/Flickr8k_Dataset/'

features = extract_features(directory)

print('Extracted Features: %d' % len(features))

# save to file

dump(features, open('/kaggle/working/features.pkl', 'wb'))
# load doc into memory

def load_doc(filename):

    # open the file as read only

    file = open(filename, 'r')

    # read all text

    text = file.read()

    # close the file

    file.close()

    return text

 

filename = '/kaggle/input/flicker8k-dataset/flickr8k_text/Flickr8k.token.txt'

# load descriptions

doc = load_doc(filename)
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

 

# parse descriptions

descriptions = load_descriptions(doc)

print('Loaded: %d ' % len(descriptions))
import string

def clean_descriptions(descriptions):

    # prepare translation table for removing punctuation

    table = str.maketrans('', '', string.punctuation)

    for key, desc_list in descriptions.items():

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

 

# clean descriptions

clean_descriptions(descriptions)
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

	table = str.maketrans('', '', string.punctuation)

	for key, desc_list in descriptions.items():

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

 

filename = '/kaggle/input/flicker8k-dataset/flickr8k_text/Flickr8k.token.txt'

# load descriptions

doc = load_doc(filename)

# parse descriptions

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

filename = '/kaggle/input/flicker8k-dataset/flickr8k_text/Flickr_8k.trainImages.txt'

train = load_set(filename)

print('Dataset: %d' % len(train))

# descriptions

train_descriptions = load_clean_descriptions('descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

# photo features

train_features = load_photo_features('/kaggle/working/features.pkl', train)

print('Photos: train=%d' % len(train_features))