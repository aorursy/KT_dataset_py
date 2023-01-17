import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import seed

from random import randint

import os

from keras.models import load_model

import pickle

import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
def createCaption(photo, model):

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

    #final = ' '.join(final)

    return final
def getDataset(path):

    f = open(path, 'r')

    fns = f.read()

    return fns.split("\n")[:-1]
#Read Img Features



infile = open("../input/flickr8k-image-extraction/img_extract.pkl",'rb')

img_fea = pickle.load(infile)

infile.close()
#Read Word Index

infile = open("../input/flickr8k-captions/word_index.pkl",'rb')

word_index = pickle.load(infile)

infile.close()



infile = open("../input/flickr8k-captions/index_word.pkl",'rb')

index_word = pickle.load(infile)

infile.close()
val_fns = getDataset("../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt")

val_set = dict((k, img_fea[k]) for k in val_fns)

max_length = 34
#256 unit LSTM, with dropout 0.20

model_3_20_256 = load_model('../input/model-256-x-20/model_256_3_20.h5')

model_6_20_256 = load_model('../input/model-256-x-20/model_256_6_20.h5')

model_10_20_256 = load_model('../input/model-256-x-20/model_256_10_20.h5')



#256 unit LSTM, with dropout 0,50

model_3_50_256 = load_model('../input/model-256-x-50/model_256_3_50.h5') 

model_6_50_256 = load_model('../input/model-256-x-50/model_256_6_50.h5')

model_10_50_256 = load_model('../input/model-256-x-50/model_256_10_50.h5')

model_20_50_256 = load_model('../input/model-256-x-50/model_256_20_50.h5') 



#512 unit LSTM, with dropout 0.20

model_3_20_512 = load_model('../input/model-512-x-20/model_512_3_20.h5')

model_6_20_512 = load_model('../input/model-512-x-20/model_512_6_20.h5')

model_10_20_512 = load_model('../input/model-512-x-20/model_512_10_20.h5')



##256 unit LSTM, with dropout 0.50

model_3_50_512 = load_model('../input/model-512-6-50/model_512_3_50.h5')

model_6_50_512 = load_model('../input/model-512-6-50/model_512_6_50.h5')

model_10_50_512 = load_model('../input/model-512-6-50/model_512_10_50.h5')



model_256_20 = [model_3_20_256, model_6_20_256, model_10_20_256]

model_256_50 = [model_3_50_256, model_6_50_256, model_10_50_256, model_20_50_256]

model_512_20 = [model_3_20_512, model_6_20_512, model_10_20_512]  

model_512_50 = [model_3_50_512, model_6_50_512, model_10_50_512]
# seed random number generator

seed(randint(0, 230692))

# generate some integers

photos = []

for _ in range(10):

    value = randint(0, len(val_fns))

    photos.append(value)
images = "../input/flickr8k/Flickr_Data/Flickr_Data/Images/"

for p in photos:

    sample_test = val_fns[p]

    sample_fea = val_set[sample_test]

    x=plt.imread(images + sample_test)

    plt.imshow(x)

    plt.show()



    conf = ""

    print("256 unit LSTM, with dropout 0.20")

    for i, model in enumerate(model_256_20):

        if i == 0:

            conf = "Picture/batch = 3: "

        elif i == 1:

            conf = "Picture/batch = 6: "

        else:

            conf = "Picture/batch = 10: "    

            

        a = createCaption((sample_fea).reshape((1,2048)), model)

        print(conf, ' '.join(a))

        

    print("\n256 unit LSTM, with dropout 0.50")

    for i, model in enumerate(model_256_50):

        if i == 0:

            conf = "Picture/batch = 3: "

        elif i == 1:

            conf = "Picture/batch = 6: "

        elif i == 2:

            conf = "Picture/batch = 10: "

        else:

            conf = "Picture/batch = 20: "

            

        a = createCaption((sample_fea).reshape((1,2048)), model)

        print(conf, ' '.join(a))

        

    print("\n512 unit LSTM, with dropout 0.20")

    for i, model in enumerate(model_512_20):

        if i == 0:

            conf = "Picture/batch = 3: "

        elif i == 1:

            conf = "Picture/batch = 6: "

        else:

            conf = "Picture/batch = 10: "

            

        a = createCaption((sample_fea).reshape((1,2048)), model)

        print(conf, ' '.join(a))

        

    print("\n512 unit LSTM, with dropout 0.50")

    for i, model in enumerate(model_512_50):

        if i == 0:

            conf = "Picture/batch = 3: "

        elif i == 1:

            conf = "Picture/batch = 6: "

        else:

            conf = "Picture/batch = 10: "

            

        a = createCaption((sample_fea).reshape((1,2048)), model)

        print(conf, ' '.join(a))