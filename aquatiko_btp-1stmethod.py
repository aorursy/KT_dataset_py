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
# from sagemaker import get_execution_role



# role = get_execution_role()

# bucket = "group24-BTP"
import numpy as np

import os

from os import path

import pandas as pd

import pickle

from pickle import dump

from keras.preprocessing import image, sequence,text

from keras.applications import inception_v3

from keras.layers import Input,Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Concatenate

from keras.models import Sequential, Model

from keras.optimizers import Adam, Nadam, SGD

from keras.applications.inception_v3 import InceptionV3

from os import listdir

from keras.preprocessing.sequence import pad_sequences

from keras.layers.merge import add

from keras.utils import to_categorical,plot_model

from keras.callbacks import ModelCheckpoint



from numpy import array

from nltk import word_tokenize
!wget https://group24-btp.s3.amazonaws.com/features_train.pkl

!wget https://group24-btp.s3.amazonaws.com/features_test.pkl

!wget https://group24-btp.s3.amazonaws.com/features_val.pkl

!wget https://group24-btp.s3.amazonaws.com/tokens.pkl

!wget https://group24-btp.s3.amazonaws.com/tok2indx.pkl

!wget https://group24-btp.s3.amazonaws.com/indx2tok.pkl

!wget https://group24-btp.s3.amazonaws.com/embedding_matrix.pkl
!wget https://group24-btp.s3.amazonaws.com/hindi-visual-genome-dev.txt

!wget https://group24-btp.s3.amazonaws.com/hindi-visual-genome-test.txt

!wget https://group24-btp.s3.amazonaws.com/hindi-visual-genome-train.txt
!ls
data_loc="hindi-visual-genome-train.txt"

train_data=pd.read_csv(data_loc,sep='\t',engine='python',header=None, nrows = 8000)

data_loc1="hindi-visual-genome-dev.txt"

val_data=pd.read_csv(data_loc1,sep='\t',engine='python',header=None)

data_loc2="hindi-visual-genome-test.txt"

test_data=pd.read_csv(data_loc2,sep='\t',engine='python',header=None)

import pickle

vocab_size= 4199

max_length=40

features_train = pickle.load(open('features_train.pkl', 'rb'))

features_test = pickle.load(open('features_test.pkl', 'rb'))

features_val = pickle.load(open('features_val.pkl', 'rb'))

tokens = pickle.load(open('tokens.pkl', 'rb'))

tok2indx = pickle.load(open('tok2indx.pkl', 'rb'))

indx2tok = pickle.load(open('indx2tok.pkl', 'rb'))

embedding_matrix = pickle.load(open('embedding_matrix.pkl', 'rb'))
tt=os.listdir("../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test")



from keras.preprocessing import image

train_images_path="../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-train.images/"

test_images_path="../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-test.images/"

val_images_path="../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-dev.images/"



from PIL import Image

import glob

train_img=glob.glob(train_images_path+'*.jpg')

test_img=glob.glob(test_images_path+'*.jpg')

val_img=glob.glob(val_images_path+'*.jpg')



train_images_path1="../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-train.images"

test_images_path1="../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-test.images"

val_images_path1="../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-dev.images"
# train_images_path="s3://group24-btp/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-train.images/"

# test_images_path="s3://group24-btp/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-test.images/"

# val_images_path="s3://group24-btp/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-dev.images/"
# from PIL import Image

# import glob

# train_img=glob.glob(train_images_path+'*.jpg')

# test_img=glob.glob(test_images_path+'*.jpg')

# val_img=glob.glob(val_images_path+'*.jpg')
# train_images_path1="s3://group24-btp/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-train.images"

# test_images_path1="s3://group24-btp/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-test.images"

# val_images_path1="s3://group24-btp/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-dev.images"
skip=[]

for index, row in val_data.iterrows():

    for i in word_tokenize(row[6]):

        if i not in tokens:

            skip.append(index)

            continue

val_data.drop(skip, inplace = True)
def load_desc(data):

        dictt=dict()

        for i in data.iterrows():

            sent=i[1][6]

            if type(sent) == str:

                sent='| '+sent+' `'

            k=str(i[1][0])

            dictt[k] = sent

            #print(k,sent)

            #dictt.update({k:sent})

        return dictt

train_desc=load_desc(train_data)

val_desc=load_desc(val_data)

test_desc=load_desc(test_data)
import nltk

nltk.download('punkt')

def create_sequences(vocab_size,max_length,descriptions,photos):

        x1,x2,y=list(),list(),list()

        for key,desc in descriptions.items():

            #print(type(desc))

            if(type(desc)!= str):   

                continue



            seq=[]

            for tok in word_tokenize(desc):

                try:

                    seq.append(tok2indx[tok])

                    for i in range(len(seq)):

                        in_seq,out_seq=seq[:i],seq[i]

                        in_seq=pad_sequences([in_seq],padding='post',maxlen=max_length,value=tok2indx[''])[0]

                        out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]

                        x1.append(photos[key][0])

                        x2.append(in_seq)

                        y.append(out_seq)

                except:

                    pass

        return array(x1),array(x2),array(y)
# from nltk.translate.bleu_score import sentence_bleu

# import keras.backend as K



# def belu_score(y_true, y_pred):

# #     true = ""

# #     pred = ""

# #     for i in y_true:

# #         true += indx2tok[i]

# #     for i in y_pred:

# #         pred += indx2tok[i]

#     print(y_true,"\n\n\n\n", y_pred)

# #     return sentence_bleu(true, pred)

#     return K.mean(y_pred)
def define_model(max_length,vocab_size, embedding_train):

        # feature extractor model

        inputs1 = Input(shape=(2048,))

        fe1 = Dropout(0.3)(inputs1)

        #extra1 = Dense(300, activation='relu')(fe1)

        fe2 = Dense(300, activation='relu')(fe1)#(extra1)

        # sequence model

        inputs2 = Input(shape=(max_length,))

        se1 = embedding_layer = Embedding(vocab_size,

                                300,

                                weights=[embedding_train],

                                input_length=40,

                                trainable=False)(inputs2)

        se2 = Dropout(0.4)(se1)

        se3 = LSTM(300)(se2)

        # decoder model

        decoder1 = add([fe2, se3])

        #extra2 = Dense(300, activation='relu')(decoder1)

        decoder2 = Dense(300, activation='relu')(decoder1)#(extra2)

        output = Dense(vocab_size, activation='softmax')(decoder2)

        # tie it together [image, seq] [word]

        model = Model(inputs=[inputs1, inputs2], output=output)

        opt= SGD(lr=0.1, momentum=0.2, decay=0, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=opt)

        # summarize model

        print(model.summary())

        #plot_model(model, to_file='model.png', show_shapes=True)

        return model
x1train,x2train,ytrain=create_sequences(vocab_size,max_length,train_desc,features_train)

x1val,x2val,yval=create_sequences(vocab_size,max_length,val_desc,features_val)
Model = define_model(max_length, vocab_size, embedding_matrix)
filepath = 'model_5-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')

Model.fit([x1train, x2train], ytrain, epochs=80, verbose=2, callbacks=[checkpoint], validation_data=([x1val, x2val], yval))    
# def extract_features1(directory):

#     model=InceptionV3()

#     model.layers.pop()

#     model=Model(inputs=model.inputs,outputs=model.layers[-1].output)

#     filename=directory+ '/' +'11.jpg'

#     img=image.load_img(filename,target_size=(299,299))

#     img=image.img_to_array(img)

#     img=np.expand_dims(img,axis=0)

#     img=inception_v3.preprocess_input(img)

#     feature=model.predict(img,verbose=0)

#     return feature

# directory='../input/hindi-visual-genome/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-10-train-dev-test/hindi-visual-genome-train.images/'   

# photo=extract_features1(directory)



# def generate_desc(model,photo,max_length):

#     seq=[]

#     seq.append(tok2indx['|'])

#     for i in range(max_length):

#         sequence1=pad_sequences([seq],padding='post',maxlen=max_length,value=tok2indx[''])

#         model.load_weights('../input/bestepoch/model_5-ep005-loss3.153-val_loss3.479.h5')

#         yhat=model.predict([photo,sequence1], verbose=0)

#         #print(len(yhat))

#         yhat=np.argmax(yhat)

#         #print(yhat)

#         word=word_for_id(yhat)

#         #print(word)

#         if word is None:

#             break

#         seq.append(tok2indx[word])

#         #print(seq)

#         if word=='`':

#             break

        

#     return seq

# numb=generate_desc(Model1,photo,max_length)

# print(numb)
# sent=""

# for i in range(len(numb)):

#     word=indx2tok2[numb[i]]

#     if i!=0:

#         sent+=' '+word

#     else:

#         sent=word



# print(sent)   

# print(train_desc['2379173'])