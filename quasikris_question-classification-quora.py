import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import word_tokenize

import re

import random

from gensim.models import KeyedVectors

import csv

import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Dropout, Embedding,LSTM, CuDNNLSTM ,ZeroPadding2D, Conv1D, MaxPooling1D, Flatten ,Input

from keras.layers import Concatenate

from keras.models import Sequential, Model

from keras.preprocessing.text import Tokenizer

from tqdm import tqdm,tqdm_notebook 

import spacy

from keras.models import load_model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.metrics import f1_score

import h5py

import gc

import operator



dftrain = pd.read_csv("../input/train.csv")

dftest = pd.read_csv("../input/test.csv")

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

spell=dict(mispell_dict)

spell.update(contraction_mapping)
train_ques=dftrain["question_text"].fillna("_##_").values

test_ques=dftest["question_text"].fillna("_##_").values

ids=dftest["qid"].fillna("_##_").values
features_nb=75000

seq_len=80

tkn=Tokenizer(lower = True, filters='', num_words=features_nb)

tkn.fit_on_texts(train_ques)

def preproc(words):

    newwords=[]

    for word in words:

        punc=0

        for p in punct:

            if word==p:

                punc=1

        if punc==0:

            word=word.lower()

            word=re.sub('[0-9]{1,}','#',word)

            for mispelling in spell.keys():

                word=word.replace(mispelling,spell[mispelling])

            newwords.append(word)

    

    return newwords





def vectorize(text):

    questions=[]

    for item in tqdm_notebook(text):   

        i=word_tokenize(item)

        i=preproc(i)

        i=' '.join(i)

        questions.append(i)

    

    seq=tkn.texts_to_sequences(questions)

    seq = pad_sequences(seq,maxlen=seq_len)

    

    return seq



train_data=vectorize(train_ques)

test_data=vectorize(test_ques)
def folds(k):

    m=len(dftrain)//5

    if(k==0):

        test=train_data[0:m]

        train=train_data[m:5*m]

        y_test=train_labels[0:m]

        y_train=train_labels[m:5*m]

    else:

        test=train_data[m*k:(k+1)*m]

        train=np.concatenate((train_data[0:m*k] , train_data[(k+1)*m:5*m]))

        y_test=train_labels[m*k:(k+1)*m]

        y_train=np.concatenate((train_labels[0:m*k] , train_labels[(k+1)*m:5*m]))

    

    return test,y_test,train,y_train
train_labels=dftrain['target'].fillna("_##_").values

test_samples,test_labels,train_samples,train_labels=folds(1)
file='../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

word2vec_index=KeyedVectors.load_word2vec_format(file, binary=True,limit=75000)

index=tkn.word_index

glove_index={}

file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

f=open(file)

k=0

for line in tqdm_notebook(f):

    components=line.split()

    word=components[0]

    vector=np.asarray(components[1:])

    if len(vector)<301 and k<features_nb:

        try:

            i=index[word]

            glove_index[word]=vector

            k+=1

        except KeyError:

            pass

    

f.close()

print(k)

del dftrain,train_data

gc.collect()
length=features_nb+1

emb_matrix=np.zeros((length,300))

for word, i in index.items():

    if i<features_nb:

        try:    

            emb_matrix[i]=np.asarray(glove_index[word],dtype='float32')*0.7 + word2vec_index[word]*0.3

        except KeyError:

            pass
print(np.shape(glove_index))
del glove_index,word2vec_index

gc.collect()
x_array=np.vstack(train_samples)



y_array=np.zeros((len(train_labels),2))



for i in range(len(train_labels)):

    if int(train_labels[i])==0:

        y_array[i]=np.array([1,0])

    else:

        y_array[i]=np.array([0,1])

   





x_validate=np.vstack(test_samples)



y_validate=np.zeros((len(test_labels),2))



for i in range (len(test_labels)):

    if int(test_labels[i])==0:

        y_validate[i]=np.array([1,0])

    else:

        y_validate[i]=np.array([0,1])
#https://www.kaggle.com/artgor/eda-and-lstm-cnn/notebook



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        """

        Keras Layer that implements an Attention mechanism for temporal data.

        Supports Masking.

        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

        # Input shape

            3D tensor with shape: `(samples, steps, features)`.

        # Output shape

            2D tensor with shape: `(samples, features)`.

        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

        The dimensions are inferred based on the output shape of the RNN.

        Example:

            model.add(LSTM(64, return_sequences=True))

            model.add(Attention())

        """

        self.supports_masking = True

        #self.init = initializations.get('glorot_uniform')

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        # eij = K.dot(x, self.W) TF backend doesn't support it



        # features_dim = self.W.shape[0]

        # step_dim = x._keras_shape[1]



        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())



        # in some cases especially in the early stages of training the sum may be almost zero

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

    #print weigthted_input.shape

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        #return input_shape[0], input_shape[-1]

        return input_shape[0],  self.features_dim
def find_best_threshold(model):

    pred_val_y = model.predict(x_validate)

    best_thresh = 0.5

    best_score = 0.0

    for thresh in np.arange(0, 1, 0.01):

        #thresh = np.round(thresh, 2)

        score = f1_score(y_validate, (pred_val_y > thresh).astype(int),average='micro')

        if score > best_score:

            best_thresh = thresh

            best_score = score

    print(best_thresh)

    print("Val F1 Score: {:.4f}".format(best_score))

    return best_thresh
models=[]

inp = Input(shape=(seq_len,))

emb = Embedding(features_nb+1,

                        300,

                        weights=[emb_matrix],

                        trainable=False,

                        input_length=seq_len)(inp)

conv1=Conv1D(32, 3, activation='relu')(emb)

max_pool1=MaxPooling1D(pool_size=2)(conv1)

conv2=Conv1D(32, 5, activation='relu')(emb)

max_pool2=MaxPooling1D(pool_size=2)(conv2)

x=Concatenate(axis=1)([max_pool1,max_pool2])

x=Flatten()(x)

x=Dropout(0.2)(x)

x=Dense(128,activation='relu')(x)

outp=Dense(2, activation='softmax')(x)

cnn = Model(inputs=inp, outputs=outp)

cnn.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
cnn.fit(x_array,y_array,epochs=5,batch_size=512,validation_data=(x_validate,y_validate))
cnn_threshold=find_best_threshold(cnn)

models.append((cnn,cnn_threshold))
from keras.layers import SpatialDropout1D , Bidirectional,CuDNNGRU,BatchNormalization



sp=SpatialDropout1D(0.3)(emb)

cgru=Bidirectional(CuDNNGRU(128,return_sequences=True))(sp)

x=Attention(seq_len)(cgru)

x=Dropout(0.2)(x)

x=Dense(128,activation='relu')(x)

x = BatchNormalization()(x)

outgru=Dense(2, activation='softmax')(x)

gru = Model(inputs=inp, outputs=outgru)

gru.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
gru.fit(x_array,y_array,epochs=5,batch_size=512,validation_data=(x_validate,y_validate))

gru_threshold=find_best_threshold(gru)

models.append((gru,gru_threshold))
from keras.layers import AveragePooling1D

z=Conv1D(32, 3, activation='relu')(cgru)

avgp=AveragePooling1D()(z)

maxp=MaxPooling1D()(z)

z=Concatenate(axis=1)([avgp,maxp])

z=BatchNormalization()(z)

z=Dropout(0.2)(z)

z=Dense(128,activation='relu')(z)

outpool=Dense(2, activation='softmax')(x)

grupool = Model(inputs=inp, outputs=outpool)

grupool.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
grupool.fit(x_array,y_array,epochs=5,batch_size=512,validation_data=(x_validate,y_validate))

pool_threshold=find_best_threshold(grupool)

models.append((grupool,pool_threshold))
from keras.layers import CuDNNLSTM



lstm = Bidirectional(CuDNNLSTM(128, return_sequences = True))(sp)

convlstm=Conv1D(32, 3, activation='relu')(lstm)

maxlstm=MaxPooling1D(pool_size=2)(convlstm)

convgru=Conv1D(32, 3, activation='relu')(cgru)

maxgru=MaxPooling1D(pool_size=2)(convgru)

x=Concatenate(axis=1)([maxlstm,maxgru])

x=Flatten()(x)

x=Dense(128,activation='relu')(x)

x=Dropout(0.2)(x)

outp=Dense(2, activation='softmax')(x)

gru_lstm = Model(inputs=inp, outputs=outp)

gru_lstm.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
gru_lstm.fit(x_array,y_array,epochs=5,batch_size=512,validation_data=(x_validate,y_validate))

lstm_threshold=find_best_threshold(gru_lstm)

models.append((gru_lstm,lstm_threshold))
gru2=Bidirectional(CuDNNGRU(64,return_sequences=True))(cgru)

x=Attention(seq_len)(gru2)

x=Dropout(0.2)(x)

x=Dense(64,activation='relu')(x)

x = BatchNormalization()(x)

outgrux2=Dense(2, activation='softmax')(x)

grux2 = Model(inputs=inp, outputs=outgrux2)

grux2.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
grux2.fit(x_array,y_array,epochs=5,batch_size=512,validation_data=(x_validate,y_validate))

grux2_threshold=find_best_threshold(grux2)

models.append((grux2,grux2_threshold))
del x_array,y_array, train_ques

gc.collect()
from statistics import mode

def results(test_samples,ques_id):

    res={}

    qid=ques_id

    thresholds=[]

    questions=np.vstack(test_samples)

    pred_list=[]

    for model in models:

        prediction_model=model[0].predict(questions)

        pred_list.append((prediction_model,model[1]))

    

    

    predictions=np.zeros(len(qid))

    for i in tqdm(range(len( qid))):

        votes=[]

        for p in pred_list:

            if p[0][i][1]>p[1]:

                votes.append(1)

            else:

                votes.append(0)

            

        md=mode(votes)

        if(md==1):

            predictions[i]=1

        else:

            predictions[i]=0

        

   

    

    for m,ids in tqdm_notebook(enumerate(qid)):

        res[ids]=predictions[m]

    

    return res 



results_dict=results(test_data,ids)
def writeOutput(results):

    header = ["qid", "prediction"]

    output_file=open("submission.csv", "w")

    writer = csv.DictWriter(output_file,fieldnames=header)

    writer.writeheader()

    

    m=0

    k=0

    

    for item in results.keys():

        if results[item]==1:

            k+=1

        else:

            m+=1

        ro={"qid":item,"prediction":int(results[item])}

        writer.writerow(ro)

    print(k)

    print(k/len(results))

    print(m)

    print(m/len(results))

    

    output_file.close() 

    



writeOutput(results_dict)
