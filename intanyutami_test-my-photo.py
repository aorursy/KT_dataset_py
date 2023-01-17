from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



import re

import time

import json

from glob import glob

from PIL import Image



import tensorflow as tf

import numpy as np

from keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input as rn_preprocess

import os



from keras.models import load_model

import pickle

import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences



from random import randint

from random import seed
#Load the extracted features by ResNet50

infile = open("../input/rn-extract/img_extract_rn50.pkl",'rb')

img_fea_rn = pickle.load(infile)

infile.close()
#Collecting all test images filenames

test_path = "../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt"

f = open(test_path, 'r')

capts = f.read()

test_photos = []



try:

    for line in capts.split("\n"):

        test_photos.append(line)

except:

    pass
#Load model 

model_rn50 = load_model('../input/model-final/model_rn50_glove.h5')



#Read Word Index

infile = open("../input/flickr8k-captions/word_index.pkl",'rb')

word_index = pickle.load(infile)

infile.close()



infile = open("../input/flickr8k-captions/index_word.pkl",'rb')

index_word = pickle.load(infile)

infile.close()
# Collecting all captions from all images, for comparison

fn = "../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"

f = open(fn, 'r')

capts = f.read()

#Group all captions by filename, for references

captions = dict()

i = 0



try:

    for line in capts.split("\n"):

        txt = line.split('\t')

        fn = txt[0].split('#')[0]

        if fn not in captions.keys():

            captions[fn] = [txt[1]]

        else:

            captions[fn].append(txt[1])

        i += 1

except:

    pass
#Create Caption

def createCaption(photo, model, max_length = 34):

    in_text = 'startseq'

    for i in range(max_length):

        sequence = [word_index[w] for w in in_text.split() if w in word_index]

        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo,sequence], verbose=0)

        yhat = np.argmax(yhat)

        word = index_word[yhat]

        in_text += ' ' + word

        if word == 'endseq':

            break

    final = in_text.split()

    final = final[1:-1]

    return ' '.join(final)
#Randomly pick 15 samples from test set

seed(randint(0,len(test_photos)))

photos = []

for _ in range(15):

    value = randint(0, len(test_photos))

    photos.append(value)
# Predict Captions from the selected test images

for p in photos:

    

    x=plt.imread("../input/flickr8k/Flickr_Data/Flickr_Data/Images/" + test_photos[p])

    plt.imshow(x)

    plt.show()

    

    sample_fea_rn = img_fea_rn[test_photos[p]]

    

    caption_rn = createCaption((sample_fea_rn).reshape((1,2048)), model_rn50)

    

    print("Real caption: ", captions[test_photos[p]][0])

    print("RN50: ", caption_rn)

    