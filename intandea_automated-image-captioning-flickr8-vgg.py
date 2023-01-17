# Load libraries

import matplotlib.pyplot as plt

import pandas as pd

import pickle

import numpy as np

import os

from keras.applications.inception_v3 import InceptionV3

# from keras.applications.resnet50 import ResNet50

# from tensorflow.keras.applications.vgg16 import VGG16

from keras.optimizers import Adam

from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate

from keras.models import Sequential, Model

from keras.utils import np_utils

import random

from keras.preprocessing import image, sequence

import matplotlib.pyplot as plt

import time
# Load data

images_dir = os.listdir("../input/flickr8k/flickr_data/Flickr_Data/")



images_path = '../input/flickr8k/flickr_data/Flickr_Data/Images/'

captions_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

train_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'

val_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'

test_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'



captions = open(captions_path, 'r').read().split("\n")

x_train = open(train_path, 'r').read().split("\n")

x_val = open(val_path, 'r').read().split("\n")

x_test = open(test_path, 'r').read().split("\n")
# Loading captions as values and images as key in dictionary

tokens = {}



for ix in range(len(captions)-1):

    temp = captions[ix].split("#")

    if temp[0] in tokens:

        tokens[temp[0]].append(temp[1][2:])

    else:

        tokens[temp[0]] = [temp[1][2:]]
# displaying an image and captions given to it

temp = captions[10].split("#")

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
# # Loading 50 layer Residual Network Model and getting the summary of the model

# # from IPython.core.display import display, HTML

# # display(HTML("""<a href="http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006">ResNet50 Architecture</a>"""))

# model = VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

# # model.summary()

# # Note: For more details on ResNet50 architecture you can click on hyperlink given below
# Helper function to process images

def preprocessing(img_path):

    im = image.load_img(img_path, target_size=(299,299,3))

    im = image.img_to_array(im)

    im = np.expand_dims(im, axis=0)

    return im
#Read Img Features

infile = open("../input/flickr8k-image-extraction/img_extract.pkl",'rb')

# infile = open("../input/image-captioning-extracting-features/img_extract_inc_3d.pkl",'rb')

img_fea = pickle.load(infile)

infile.close()
train_data = dict((k, img_fea[k]) for k in x_train[:-1])
# train_data = {}

# ctr=0

# for ix in x_train:

#     if ix == "":

#         continue

#     if ctr >= 3000:

#         break

#     ctr+=1

#     if ctr%1000==0:

#         print(ctr)

#     path = images_path + ix

#     img = preprocessing(path)

#     pred = model.predict(img).reshape(2048)

#     train_data[ix] = pred
train_data['2513260012_03d33305cf.jpg'].shape
# # opening train_encoded_images.p file and dumping it's content

# with open( "train_encoded_images.p", "wb" ) as pickle_f:

#     pickle.dump(train_data, pickle_f )  
# Loading image and its corresponding caption into a dataframe and then storing values from dataframe into 'ds'

pd_dataset = pd.read_csv("flickr_8k_train_dataset.txt", delimiter='\t')

ds = pd_dataset.values

print(ds.shape)
pd_dataset.head()
pd_dataset.captions = [item.lower() for item in pd_dataset.captions]

pd_dataset.captions = pd_dataset.captions.apply(lambda x: x.replace('.', ''))
# Storing all the captions from ds into a list

sentences = []

for ix in range(ds.shape[0]):

    sentences.append(ds[ix, 1])

    

print(len(sentences))
# First 5 captions stored in sentences

sentences[:5]
# Splitting each captions stored in 'sentences' and storing them in 'words' as list of list

# words = [i.split() for i in sentences]
sentences_low = [item.lower() for item in sentences]
from keras.preprocessing.text import Tokenizer



#Tokenize top 5000 words in Train Captions

tokenizer = Tokenizer(num_words=5000,

                      oov_token="<unk>",

                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer.fit_on_texts(sentences_low)

word_2_indices = tokenizer.word_index

indices_2_word = tokenizer.index_word
pickle.dump(word_2_indices ,open("word_2_indices.pkl","wb"))

pickle.dump(indices_2_word ,open("indices_2_word.pkl","wb"))
vocab_size = len(word_2_indices.keys())

print(vocab_size)
# # Creating a list of all unique words

# unique = []

# for i in words:

#     unique.extend(i)

# unique = list(set(unique))



# print(len(unique))



# vocab_size = len(unique)
# # Vectorization

# word_2_indices = {val:index for index, val in enumerate(unique)}

# indices_2_word = {index:val for index, val in enumerate(unique)}
# word_2_indices['UNK'] = 0

# word_2_indices['raining'] = 8253
# indices_2_word[0] = 'UNK'

# indices_2_word[8253] = 'raining'
# print(word_2_indices['<start>'])

# print(indices_2_word[4011])

# print(word_2_indices['<end>'])

# print(indices_2_word[8051])
max_len = 0



for i in sentences_low:

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
# print(padded_sequences[0])
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
# with open('../input/train_encoded_images.p', 'rb') as f:

#     encoded_images = pickle.load(f, encoding="bytes")
# ds[1, 0].encode()
imgs = []



for ix in range(ds.shape[0]):

    if ds[ix, 0] in train_data.keys():

#         print(ix, encoded_images[ds[ix, 0].encode()])

        imgs.append(list(train_data[ds[ix, 0]]))



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
hist = model.fit([images, captions], next_words, batch_size=512, epochs=150)
model.save_weights("model_weights_inc_v3.h5")
def preprocessing(img_path):

    im = image.load_img(img_path, target_size=(224,224,3))

    im = image.img_to_array(im)

    im = np.expand_dims(im, axis=0)

    return im
def get_encoding(model, img):

    image = preprocessing(img)

    pred = model.predict(image).reshape(2048)

    return pred
# resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
img = "../input/flickr8k/flickr_data/Flickr_Data/Images/1453366750_6e8cf601bf.jpg"



# test_img = get_encoding(resnet, img)

test_img = img_fea["1453366750_6e8cf601bf.jpg"]
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
z = Image(filename=img)

display(z)



print(Argmax_Search)