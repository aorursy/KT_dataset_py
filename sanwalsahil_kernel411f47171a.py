# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(os.path.join(dirname))

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/working'):

    print(os.path.join(dirname))

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pickle

import numpy as np

import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.models import Model

import string

import re
def extract_features(directory):

    #load the model

    model = VGG16()

    #restructure the model

    model.layers.pop()

    model = Model(inputs=model.inputs,

                 outputs=model.layers[-1].output)

    

    #extract features from each model

    features = dict()

    for name in os.listdir(directory):

        # load image from a file

        filename = directory+'/'+name

        image = load_img(filename,target_size=(224,224))

        #convert the image pixels to numpy array

        image = img_to_array(image)

        #reshape data for the model

        image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))

        #prepare the image for the vgg model

        image = preprocess_input(image)

        #get features

        feature = model.predict(image)

        # get image id

        image_id = name.split('.')[0]

        #store feature

        features[image_id] = feature

    return features
#imageDir = '../input/flickr8k/flickr_data/Flickr_Data/Images'

#features = extract_features(imageDir)
#pickle.dump(features, open('features.pkl','wb'))
from IPython.display import FileLink

FileLink('features.pkl')
def load_doc(filename):

    #open the filename as readonly

    file = open(filename,'r')

    #read all text

    text = file.read()

    #close the file

    file.close()

    return text
filename = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'
doc = load_doc(filename)
doc[:100]
#features['1000268201_693b08cb0e']
#extract descriptions for images

def load_descriptions(doc):

    mapping = dict()

    #process lines

    for line in doc.split('\n'):

        #split line by white space

        tokens = line.split()

        if len(line)<2:

            continue

        #take first token as image id and rest as description

        image_id,image_desc = tokens[0],tokens[1:]

        #remove filename from image id

        image_id = image_id.split('.')[0]

        # convert description tokens back to strings

        image_desc = ' '.join(image_desc)

        # create the list

        if image_id not in mapping:

            mapping[image_id] = list()

        #store description

        mapping[image_id].append(image_desc)

    return mapping
descriptions = load_descriptions(doc)
descriptions
def clean_descriptions(descriptions):

    # prepare regex for char filtering

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))

    for key,desc_list in descriptions.items():

        for i in range(len(desc_list)):

            desc = desc_list[i]

            #tokenize

            desc = desc.split()

            #convert to lowercase

            desc = [word.lower() for word in desc]

            #remove puntuation from each token

            desc = [re_punc.sub('',w) for w in desc]

            #remove single letter words

            desc = [word for word in desc if len(word) > 1]

            #remove tokens woth numbers in them

            desc = [word for word in desc if word.isalpha()]

            #store as string

            desc_list[i] = ' '.join(desc)
clean_descriptions(descriptions)
# convert the loaded descriptions into a vocabulary of words

def to_vocabulary(descriptions):

    # build a list of all description strings

    all_desc = set()

    for key in descriptions.keys():

        [all_desc.update(d.split()) for d in descriptions[key]]

    return all_desc

# summarize vocabulary

vocabulary = to_vocabulary(descriptions)

print('Vocabulary Size: %d' % len(vocabulary))

        
#save descriptions to a file

def save_doc(descriptions,filename):

    lines = list()

    for key, desc_list in descriptions.items():

        for desc in desc_list:

            lines.append(key + ' ' + desc)

    data = '\n'.join(lines)

    file= open(filename,'w')

    file.write(data)

    file.close()
save_doc(descriptions,'descriptions.txt')
descriptions
#load a predefined list of photo identifiers

def load_set(filename):

    #get file content

    doc = load_doc(filename)

    #create empty dataset list

    dataset= list()

    #process each filename by traversing document line by line

    for line in doc.split('\n'):

        #skipe empty lines

        if len(line)<1:

            continue

        #get the image identifier

        identifier = line.split('.')[0]

        dataset.append(identifier)

    return dataset
#load clean description into memory

def load_clean_descriptions(filename,dataset):

    #load document

    doc = load_doc(filename)

    descriptions = dict()

    for line in doc.split('\n'):

        #split line by white space

        tokens = line.split()

        #split id from description

        image_id,image_desc = tokens[0],tokens[1:]

        #skip images not in the set

        if image_id in dataset:

            #create list

            if image_id not in descriptions:

                descriptions[image_id] = list()

            #wrap descriptions in tokens

            desc = 'startseq'+' '.join(image_desc)+' endseq'

            #store

            descriptions[image_id].append(desc)

    return descriptions
# load photo features

def load_photo_features(filename, dataset):

    # load all features

    all_features = pickle.load(open(filename, 'rb'))

    # filter features

    features = {k: all_features[k] for k in dataset}

    return features
trainFile = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'

train = load_set(trainFile)

print(len(train))
train = train[:3000]
train_descriptions = load_clean_descriptions('/kaggle/working/descriptions.txt',train)
print(len(train_descriptions))
# photo features

train_features = load_photo_features('/kaggle/input/features/features.pkl', train)

print('Photos: train=%d' % len(train_features))
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
#convert a dictionary of clean descriptions into a list of descriptions

def to_lines(descriptions):

    all_desc = list()

    for key in descriptions.keys():

        [all_desc.append(d) for d in descriptions[key]]

    return all_desc

#fit a tokenizer given caption descirptions

def create_tokenizer(descriptions):

    lines= to_lines(descriptions)

    tokenizer =Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer
tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) +1

print(vocab_size)
def create_sequences(tokenizer, max_length, descriptions, photos):

    X1, X2, y = list(), list(), list()

    # walk through each image identifier

    for key, desc_list in descriptions.items():

        #walk through each description for image

        for desc in desc_list:

            

            #encode the sequence

            seq = tokenizer.texts_to_sequences([desc])[0]

            #split one sequence into multiple X,y pairs

            for i in range(1,len(seq)):

                # split into input and output pair

                in_seq, out_seq = seq[:i], seq[i]

                # pad input sequence

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                # encode output sequence

                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                # store

                X1.append(photos[key][0])

                X2.append(in_seq)

                y.append(out_seq)

    return np.array(X1),np.array(X2),np.array(y)
# define the captioning model

def define_model(vocab_size, max_length):

    # feature extractor model

    inputs1 = Input(shape=(1000,))

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

    # compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model

    model.summary()

    plot_model(model, to_file='model.png', show_shapes=True)

    return model
# define checkpoint callback

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1,

save_best_only=True, mode='min')
# calculate the length of the description with the most words

def max_length(descriptions):

    lines = to_lines(descriptions)

    return max(len(d.split()) for d in lines)
# prepare tokenizer

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length

max_length = max_length(train_descriptions)

print( max_length)

len(train_descriptions)
import itertools

train_descriptions = dict(itertools.islice(train_descriptions.items(), 3000))
train_features = dict(itertools.islice(train_features.items(), 3000))
# prepare sequences

X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions,train_features)
#train_data = load_doc('../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt')
model = define_model(vocab_size,max_length)
model.fit([X1train,X2train],ytrain,epochs=20)
model.save('model.h5') 



from IPython.display import FileLink

FileLink('model.h5')
pickle.dump(tokenizer, open('tokenizer.pkl','wb'))


FileLink('tokenizer.pkl')