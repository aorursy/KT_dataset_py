import os

import time

import progressbar

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

import keras

import sys, time, os, warnings 

import numpy as np

import pandas as pd 

from collections import Counter 

from keras.preprocessing.image import load_img

from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")



## The location of the caption file

#dir_Flickr_text = "../input/flickr-image-dataset/flickr30k_images/flickr30k_images/results.csv"

#dir_Flickr_jpg = "../input/flickr-image-dataset/flickr30k_images/flickr30k_images/flickr30k_images"

dir_Flickr_text = "../input/flickr8k-sau/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"

dir_Flickr_jpg = "../input/flickr8k-sau/flickr8k-sau/Flickr_Data/Images"



jpgs = os.listdir(dir_Flickr_jpg)

print("The number of jpg flies in Flicker30k: {}".format(len(jpgs)))
## loading as dataframe

def load_csv(directory):

    desc=dict()

    text = pd.read_csv(directory, delimiter='|',header=None,names=["filename","index","caption"])

    text = text.iloc[1:,:]

    df_new = text[text.iloc[:,2].notnull()]

    print(df_new.iloc[:5,:])

    return df_new  




file = open(dir_Flickr_text,'r')

text = file.read()

file.close()





datatxt = []

for line in text.split('\n'):

    col = line.split('\t')

    if len(col) == 1:

        continue

    w = col[0].split("#")

    datatxt.append(w + [col[1].lower()])



df_txt = pd.DataFrame(datatxt,columns=["filename","index","caption"])





uni_filenames = np.unique(df_txt.filename.values)

print("The number of unique file names : {}".format(len(uni_filenames)))

print("The distribution of the number of captions for each image:")

Counter(Counter(df_txt.filename.values).values())



from keras.preprocessing.image import load_img, img_to_array



npic = 5

npix = 224

target_size = (npix,npix,3)



count = 1

fig = plt.figure(figsize=(10,20))

for jpgfnm in uni_filenames[:npic]:

    filename = dir_Flickr_jpg + '/' + jpgfnm

    captions = list(df_txt["caption"].loc[df_txt["filename"]==jpgfnm].values)

    image_load = load_img(filename, target_size=target_size)

    

    ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])

    ax.imshow(image_load)

    count += 1

    

    ax = fig.add_subplot(npic,2,count)

    plt.axis('off')

    ax.plot()

    ax.set_xlim(0,1)

    ax.set_ylim(0,len(captions))

    for i, caption in enumerate(captions):

        ax.text(0,i,caption,fontsize=20)

    count += 1

plt.show()
def df_word(df_txt):

    vocabulary = []

    for i in range(len(df_txt)):

        temp=df_txt.iloc[i,2]

        vocabulary.extend(temp.split())

    print('Vocabulary Size: %d' % len(set(vocabulary)))

    ct = Counter(vocabulary)

    dfword = pd.DataFrame({"word":list(ct.keys()),"count":list(ct.values())})

    dfword = dfword.sort_values("count",ascending=False)

    dfword = dfword.reset_index()[["word","count"]]

    return(dfword)

dfword = df_word(df_txt)

dfword.head(3)
topn = 50



def plthist(dfsub, title="The top 50 most frequently appearing words"):

    plt.figure(figsize=(20,3))

    plt.bar(dfsub.index,dfsub["count"])

    plt.yticks(fontsize=20)

    plt.xticks(dfsub.index,dfsub["word"],rotation=90,fontsize=20)

    plt.title(title,fontsize=20)

    plt.show()



plthist(dfword.iloc[:topn,:],

        title="The top 50 most frequently appearing words")

plthist(dfword.iloc[-topn:,:],

        title="The least 50 most frequently appearing words")
import string

def remove_punctuation(text_original):

    text_no_punctuation = text_original.translate(str.maketrans('','',string.punctuation))

    return(text_no_punctuation)



def remove_single_character(text):

    text_len_more_than1 = ""

    for word in text.split():

        if len(word) > 1:

            text_len_more_than1 += " " + word

    return(text_len_more_than1)



def remove_numeric(text,printTF=False):

    text_no_numeric = ""

    for word in text.split():

        isalpha = word.isalpha()

        if printTF:

            print("    {:10} : {:}".format(word,isalpha))

        if isalpha:

            text_no_numeric += " " + word

    return(text_no_numeric)

def text_clean(text_original):

    text = remove_punctuation(text_original)

    text = remove_single_character(text)

    text = remove_numeric(text)

    return(text)



with progressbar.ProgressBar(max_value=len(df_txt.caption.values)) as bar:

    for i, caption in enumerate(df_txt.caption.values):

        newcaption = text_clean(caption)

        df_txt["caption"].iloc[i] = newcaption

        bar.update(i)

    
dfword = df_word(df_txt)

plthist(dfword.iloc[:topn,:],

        title="The top 50 most frequently appearing words")

plthist(dfword.iloc[-topn:,:],

        title="The least 50 most frequently appearing words")
from copy import copy

def add_start_end_seq_token(captions):

    caps = []

    for txt in captions:

        txt = 'startseq ' + txt + ' endseq'

        caps.append(txt)

    return(caps)

df_txt0 = copy(df_txt)

df_txt0["caption"] = add_start_end_seq_token(df_txt["caption"])

df_txt0.head(5)

del df_txt

from keras.applications import VGG16



modelvgg = VGG16(include_top=True,weights=None)

## load the locally saved weights 

modelvgg.load_weights("../input/vgg16-weights-image-captioning/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

modelvgg.summary()
from keras import models

modelvgg.layers.pop()

modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)

## show the deep learning model

modelvgg.summary()
from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import preprocess_input

from collections import OrderedDict



images = OrderedDict()

npix = 224

target_size = (npix,npix,3)

with progressbar.ProgressBar(max_value=len(jpgs)) as bar:

    for i,name in enumerate(jpgs):

        # load an image from file

        filename = dir_Flickr_jpg + '/' + name

        image = load_img(filename, target_size=target_size)

        # convert the image pixels to a numpy array

        image = img_to_array(image)

        nimage = preprocess_input(image)

        y_pred = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))

        images[name] = y_pred.flatten()

        bar.update(i)

    #print(i,filename)
dimages, keepindex = [],[]

nd=(df_txt0["index"].values)

b = [(int(i)==0) for i in nd]

#for i in nd:

 #   print(int(i)==0)

#df_txt0 = df_txt0.loc[b,: ]

df_txt0 = df_txt0.loc[df_txt0["index"].values == "0",: ]



for i, fnm in enumerate(df_txt0.filename):

    if fnm in images.keys():

        dimages.append(images[fnm])

        keepindex.append(i)

        

fnames = df_txt0["filename"].iloc[keepindex].values

dcaptions = df_txt0["caption"].iloc[keepindex].values

dimages = np.array(dimages)

print(df_txt0["index"][:5])
from keras.preprocessing.text import Tokenizer

## the maximum number of words in dictionary

count_words=22000

#nb_words = 31782

tokenizer = Tokenizer(num_words=8000)

tokenizer.fit_on_texts(dcaptions)

vocab_size = len(tokenizer.word_index) + 1

print("vocabulary size : {}".format(vocab_size))

dtexts = tokenizer.texts_to_sequences(dcaptions)

print(dtexts[:5])
prop_test, prop_val = 0.2, 0.2 



N = len(dtexts)

Ntest, Nval = int(N*prop_test), int(N*prop_val)



def split_test_val_train(dtexts,Ntest,Nval):

    return(dtexts[:Ntest], 

           dtexts[Ntest:Ntest+Nval],  

           dtexts[Ntest+Nval:])



dt_test,  dt_val, dt_train   = split_test_val_train(dtexts,Ntest,Nval)

di_test,  di_val, di_train   = split_test_val_train(dimages,Ntest,Nval)

fnm_test,fnm_val, fnm_train  = split_test_val_train(fnames,Ntest,Nval)
maxlen = np.max([len(text) for text in dtexts])
from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



def preprocessing(dtexts,dimages):

    N = len(dtexts)

    print("# captions/images = {}".format(N))



    assert(N==len(dimages))

    Xtext, Ximage, ytext = [],[],[]

    for text,image in zip(dtexts,dimages):



        for i in range(1,len(text)):

            in_text, out_text = text[:i], text[i]

            in_text = pad_sequences([in_text],maxlen=maxlen).flatten()

            out_text = to_categorical(out_text,num_classes = vocab_size)



            Xtext.append(in_text)

            Ximage.append(image)

            ytext.append(out_text)



    Xtext  = np.array(Xtext)

    Ximage = np.array(Ximage)

    ytext  = np.array(ytext)

    print(" {} {} {}".format(Xtext.shape,Ximage.shape,ytext.shape))

    return(Xtext,Ximage,ytext)





Xtext_train, Ximage_train, ytext_train = preprocessing(dt_train,di_train)

Xtext_val,   Ximage_val,   ytext_val   = preprocessing(dt_val,di_val)

# pre-processing is not necessary for testing data

#Xtext_test,  Ximage_test,  ytext_test  = preprocessing(dt_test,di_test)
from keras import layers

print(vocab_size)

## image feature



dim_embedding = 64



input_image = layers.Input(shape=(Ximage_train.shape[1],))

fimage = layers.Dense(256,activation='relu',name="ImageFeature")(input_image)

## sequence model

input_txt = layers.Input(shape=(maxlen,))

ftxt = layers.Embedding(vocab_size,dim_embedding, mask_zero=True)(input_txt)

ftxt = layers.LSTM(256,name="CaptionFeature")(ftxt)

## combined model for decoder

decoder = layers.add([ftxt,fimage])

decoder = layers.Dense(256,activation='relu')(decoder)

output = layers.Dense(vocab_size,activation='softmax')(decoder)

model = models.Model(inputs=[input_image, input_txt],outputs=output)



model.compile(loss='categorical_crossentropy', optimizer='adam')



print(model.summary())
start = time.time()

#checkpoint_path = "training_1/cp.ckpt"

#checkpoint_dir = os.path.dirname(checkpoint_path)



# Create checkpoint callback

#cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 

 #                                                save_weights_only=True,

  #                                               verbose=2)



hist = model.fit([Ximage_train, Xtext_train], ytext_train, 

                  epochs=7, verbose=2, 

                  batch_size=64,

                  validation_data=([Ximage_val, Xtext_val], ytext_val))

                #callbacks = [cp_callback])

end = time.time()

print("TIME TOOK {:3.2f}MIN".format((end - start )/60))
print(Ximage_train.shape,Xtext_train.shape,ytext_train.shape)
for label in ["loss","val_loss"]:

    plt.plot(hist.history[label],label=label)

plt.legend()

plt.xlabel("epochs")

plt.ylabel("loss")

plt.show()
index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])

def predict_caption(image):

    '''

    image.shape = (1,4462)

    '''



    in_text = 'startseq'



    for iword in range(maxlen):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence],maxlen)

        yhat = model.predict([image,sequence],verbose=0)

        yhat = np.argmax(yhat)

        newword = index_word[yhat]

        in_text += " " + newword

        if newword == "endseq":

            break

    return(in_text)







npic = 5

npix = 224

target_size = (npix,npix,3)



count = 1

fig = plt.figure(figsize=(10,20))

for jpgfnm, image_feature in zip(fnm_test[:npic],di_test[:npic]):

    ## images 

    filename = dir_Flickr_jpg + '/' + jpgfnm

    image_load = load_img(filename, target_size=target_size)

    ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])

    ax.imshow(image_load)

    count += 1



    ## captions

    caption = predict_caption(image_feature.reshape(1,len(image_feature)))

    ax = fig.add_subplot(npic,2,count)

    plt.axis('off')

    ax.plot()

    ax.set_xlim(0,1)

    ax.set_ylim(0,1)

    ax.text(0,0.5,caption,fontsize=20)

    count += 1



plt.show()
from nltk.translate.bleu_score import sentence_bleu

index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])





nkeep = 5

pred_good, pred_bad, bleus = [], [], [] 

count = 0 

for jpgfnm, image_feature, tokenized_text in zip(fnm_test,di_test,dt_test):

    count += 1

    if count % 200 == 0:

        print("  {:4.2f}% is done..".format(100*count/float(len(fnm_test))))

    

    caption_true = [ index_word[i] for i in tokenized_text ]     

    caption_true = caption_true[1:-1] ## remove startreg, and endreg

    ## captions

    caption = predict_caption(image_feature.reshape(1,len(image_feature)))

    caption = caption.split()

    caption = caption[1:-1]## remove startreg, and endreg

    

    bleu = sentence_bleu([caption_true],caption)

    bleus.append(bleu)

    if bleu > 0.7 and len(pred_good) < nkeep:

        pred_good.append((bleu,jpgfnm,caption_true,caption))

    elif bleu < 0.3 and len(pred_bad) < nkeep:

        pred_bad.append((bleu,jpgfnm,caption_true,caption))


def plot_images(pred_bad):

    def create_str(caption_true):

        strue = ""

        for s in caption_true:

            strue += " " + s

        return(strue)

    npix = 224

    target_size = (npix,npix,3)    

    count = 1

    fig = plt.figure(figsize=(10,20))

    npic = len(pred_bad)

    for pb in pred_bad:

        bleu,jpgfnm,caption_true,caption = pb

        ## images 

        filename = dir_Flickr_jpg + '/' + jpgfnm

        image_load = load_img(filename, target_size=target_size)

        ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])

        ax.imshow(image_load)

        count += 1



        caption_true = create_str(caption_true)

        caption = create_str(caption)

        

        ax = fig.add_subplot(npic,2,count)

        plt.axis('off')

        ax.plot()

        ax.set_xlim(0,1)

        ax.set_ylim(0,1)

        ax.text(0,0.7,"true:" + caption_true,fontsize=20)

        ax.text(0,0.4,"pred:" + caption,fontsize=20)

        ax.text(0,0.1,"BLEU: {}".format(bleu),fontsize=20)

        count += 1

    plt.show()



print("Bad Caption")

plot_images(pred_bad)

print("Good Caption")

plot_images(pred_good)

 