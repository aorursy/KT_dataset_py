import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import word_tokenize

import re

import random

#from gensim.models import word2vec

import csv

import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Dropout, Embedding,LSTM, CuDNNLSTM ,ZeroPadding2D, Conv1D, MaxPooling1D

from keras.models import Sequential, Model

from keras.preprocessing.text import Tokenizer

from keras.models import model_from_json

from tqdm import tqdm,tqdm_notebook 

import spacy

from keras.models import load_model

import h5py



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

spell=dict(mispell_dict)

spell.update(contraction_mapping)
question_list=[]



for index,row in tqdm_notebook(train.iterrows()):

    question_list.append([row['question_text'],int(row['target'])])

    

def preproc(words):

    newwords=[]

    for word in words:

        punc=0

        for p in punct:

            if word==p:

                punc=1

        if punc==0:

            word=re.sub('[0-9]{1,}','#',word)

            for mispelling in spell.keys():

                word=word.replace(mispelling,spell[mispelling])

            newwords.append(word)

    

    return newwords



      



def vectorize(text,t):

    tokenized_questions=[]

    for item in tqdm_notebook(text):

        if t==0:

            i=item[0]

        else:

            i=item

           

        i=word_tokenize(i)

        i=preproc(i)

        i=' '.join(i)

        #i=glove(i).vector

        

        if t==0:

            tokenized_questions.append([i,item[1]])

        else:

            tokenized_questions.append(i)

    

        

    return tokenized_questions





train_data=vectorize(question_list,0)



random.shuffle(train_data)

def folds(k):

    m=len(train_data)//5

    if(k==0):

        test=train_data[0:m]

        train=train_data[m:5*m]

    else:

        test=train_data[m*k:(k+1)*m]

        train=train_data[0:m*k] + train_data[(k+1)*m:5*m]

   

    return test,train



test_samples,train_samples=folds(2)


def build_vocab(text):

    vocabulary={}

    ttl_docs=0

    for question in tqdm_notebook(text):

        ttl_docs+=1

        q=word_tokenize(question[0])

        temp={}

        for word in q:

            

            try:

                temp[word]

            except:

                try:

                    vocabulary[word] +=1

                except KeyError:

                    vocabulary[word] =1

    return ttl_docs,vocabulary





total,vocab=build_vocab(train_data)



import operator

l=sorted(vocab.items(), key=operator.itemgetter(1),reverse=True)

feat_vocab=l[:50000]



feat_vocab=dict(feat_vocab)



glove_index={}

file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

found=0

f=open(file)

for line in tqdm_notebook(f):

    components=line.split()

    word=components[0]

    vector=np.asarray(components[1:])

    if len(vector)<301:

        try:

            feat_vocab[word]

            found+=1

            glove_index[word]=vector

        except KeyError:

            pass

    

f.close()
from math import log10

def tf_idf(question):

    words={}

    for word in question:

        try:

            words[word] +=1

        except KeyError:

            words[word] =1

    

    vec=np.zeros(300, dtype='float64')

    for item in words.keys():

        try:

            weight= words[item]*log10(total/feat_vocab[item])

           

            vec=np.add(vec,glove_index[item].astype(np.float)*weight)

        except KeyError:

            pass

        

    return vec



x_train=[]

y_train=[]



for row in tqdm_notebook(train_samples):

    x_train.append(tf_idf(row[0]))

    y_train.append(row[1])
del train_samples,train ,train_data,question_list
x_array=np.vstack(x_train)

x_array = np.expand_dims(x_array, axis=2)



"""y_array=np.array(y_train"""

y_array=np.zeros((len(y_train),2))



for i in range(len(y_train)):

    if y_train[i]==0:

        y_array[i]=np.array([1,0])

    else:

        y_array[i]=np.array([0,1])

   



xs=[]

y_val=[]





for row in tqdm_notebook(test_samples):

    

    xs.append(tf_idf(row[0]))

    y_val.append(row[1]) 





x_validate=np.vstack(xs)

x_validate = np.expand_dims(x_validate, axis=2)



"""y_validate=np.array(y_val)"""



y_validate=np.zeros((len(y_val),2))



for i in range (len(y_val)):

    if y_val[i]==0:

        y_validate[i]=np.array([1,0])

    else:

        y_validate[i]=np.array([0,1])

del test_samples
from keras.layers import Flatten

cnn=Sequential()



cnn.add(Conv1D(16, 3, activation='relu',input_shape=(x_array.shape[1],1)))

cnn.add(Conv1D(16, 3, activation='relu'))

cnn.add(Conv1D(16, 3, activation='relu'))

cnn.add(MaxPooling1D(pool_size=2))

cnn.add(Flatten())

cnn.add(Dense(128,activation='relu'))

layer=Dense(100, activation='relu')

cnn.add(layer)

cnn.add(Dense(2, activation='softmax'))

cnn.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
cnn.fit(x_array,y_array,epochs=5,batch_size=128,validation_data=(x_validate,y_validate))
cnn.save("cnn.h5")
feature_layer = Model(inputs=cnn.input,outputs=layer.output)


batch_size=100000

rng=len(x_array)//batch_size + 1

def predict(array,batch,units):

    for m in range(units):

        start=m*batch

        end=(m+1)*batch

        if m==0:

            new_features=feature_layer.predict(array[:end])

        elif m==units :

            new_features=np.concatenate((new_features,feature_layer.predict(array[start:])))

        else:

            new_features=np.concatenate((new_features,feature_layer.predict(array[start:end])))

    

    return new_features
del vocab

del x_train,y_train
features_fit=predict(x_array,batch_size,rng)
del x_array
iters=len(x_validate)//batch_size + 1

features_validate=predict(x_validate,batch_size,iters)
import xgboost as xgb



dtrain=xgb.DMatrix(features_fit, label=y_array)

dvalidate=xgb.DMatrix(features_validate,label=y_validate)
del features_fit , features_validate


#for i in range(1,6):

prmtrs = {'objective': 'binary:logistic',

          'max_delta_step':1,

          'max_depth':10,

          'min_child_weight':8,

          'scale_pos_weigh':1,

          'subsample':0.55,

          'reg_alpha':0.001,

          'learning_rate':0.1

           }

prmtrs['eval_metric'] = 'auc'

dataset = [(dvalidate, 'eval'), (dtrain, 'train')]



epochs=10

xgbmodel=xgb.train(prmtrs,dtrain,epochs,dataset)

print("----------------------------------------")

print('\n')
del dtrain,dvalidate
import matplotlib

xgb.plot_importance(xgbmodel)
xgbmodel.save_model('xbgm.model')

xgbmodel.dump_model('dump.raw.txt')
"""import xgboost as xgb

loaded_model = load_model('xgbm.model')"""
import gc



gc.collect()

def results(test_samples):

    qid=[]

    questions=[]

    res={}

    for index,row in test_samples.iterrows():

        qid.append(row[0])

        questions.append(row[1])

    questions=vectorize(questions,1)

    glove=[tf_idf(q) for q in questions]

    questions=np.vstack(glove)

    questions = np.expand_dims(questions, axis=2)

    it=len(questions)//batch_size + 1

    features=predict(questions,batch_size,it)

    data=xgb.DMatrix(features)

    predictions=xgbmodel.predict(data)

    



    for m,ids in tqdm_notebook(enumerate(qid)):

        res[ids]=predictions[m]

    

    return res



results_dict=results(test)
del test


def writeOutput(results):

    header = ["qid", "prediction"]

    output_file=open("submission.csv", "w")

    writer = csv.DictWriter(output_file,fieldnames=header)

    writer.writeheader()

    

    m=0

    k=0

    

    for item in results.keys():

        if results[item]>0.501:

            value=1

            k+=1

        else:

            value=0

            m+=1

        ro={"qid":item,"prediction":value}

        writer.writerow(ro)

    print(k)

    print(k/len(results))

    print(m)

    print(m/len(results))

    

    output_file.close() 

    



writeOutput(results_dict)

print(len(results_dict))