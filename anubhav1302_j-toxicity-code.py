#Jigsaw Comment Toxicity Dataset

#In every dataset, most important part of modelling is cleaning of dataset.

#One simply has to follow some steps along with some intincts of at how much depth you want to clean the data.

#Talking about this dataset, Cleaning process is divided into 3 functions.

#1: Clean APPOSTOPHES

#2: Remove stop words along with lemmatization

#3: Removes punctuations along along with word 'haha' because it won't contribute in accuracy or loss.

#4: Last function is just to call the last function which later will call other features.

#In this cleaning process, i didn't remove rare/most common occuring words since comments tend to have only one word and 

#if that word is rare or most common and got removed then we will have null observation due to which we will have to deal with null 

#observation as well and also, some cursing/Toxic words are rare so model needs to know them.

#Load Required Libraries

import re

import gensim

import itertools

import numpy as np

import pandas as pd

from tqdm import tqdm

from keras import callbacks

from collections import Counter

from nltk.corpus import stopwords

from keras.optimizers import RMSprop,Adam

from keras.models import Model,Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from nltk.stem import WordNetLemmatizer,PorterStemmer

from keras.optimizers import Adam,RMSprop

from keras.layers import CuDNNLSTM,CuDNNGRU,Dropout,LeakyReLU,Input,Embedding,Dense,Bidirectional

#Data path

train_path='../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'

test_path='../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
#load data

train=pd.read_csv(train_path)

test=pd.read_csv(test_path)
import os 

os.listdir('../input/glove840b300dtxt')
#Load Word Embeddings

embedding_path='../input/glove840b300dtxt/glove.840B.300d.txt'

embedding_dict={}

embd_file=open(embedding_path,'r',errors = 'ignore',encoding='utf8')

for line in tqdm(embd_file):

    values=line.split(' ')

    word=values[0]

    coef=np.asarray(values[1:],dtype='float32')

    embedding_dict[word]=coef

embd_file.close()

print('Found %s word vectors.' % len(embedding_dict))
#Lets take a look at train data

train.head(50)
train_columns=train.columns

train_columns
#Lets Drop unnecessary features

uns_features=['id','created_date','publication_id','parent_id','article_id']

train=train.drop(uns_features,axis=1)
#Since target is in Probability form so, we will transform it into class form

#As described in data description, target>=0.5 is toxic/positive class.

train['target'][train['target']>=0.5]=1

train['target'][train['target']<0.5]=0
train['target'].value_counts()

train_labels=train['target']
#Now lets look at our comment section

train['comment_text'].head()

#We are going to remove unnecessary stopwords,punctuations and numbers in text if present.
#def text_cleaner_1(text): #Correct APPOSTOPHES

#    words=re.split(r'\W+',text)

#    APPOSTOPHES= {"s": "is", "re":"are","ll":"will"} 

#    tmp=[APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]

#    return ' '.join(tmp)



def text_cleaner_2(text): #Remove StopWords and Lemmatize words

    tmp=[]

    #text=text_cleaner_1(text)

    lm=WordNetLemmatizer()

    ps=PorterStemmer()

    sp=stopwords.words('english')

    for word in text.split():

        if word not in sp:

            tmp.append(lm.lemmatize(word.lower()))

        

    return ' '.join(tmp)



def text_cleaner_3(text): #Remove punctuations

    text=text_cleaner_2(text)

    text=re.sub(r'[^\w\s]','',text)

    text=re.sub(r'h+a+h+a','',text) #Remove haha/hahaha...

    text=re.sub(r'f+f+u+u+','fuck',text) #lol :D

    text =''.join(''.join(s)[:2] for _,s in itertools.groupby(text))

    return text



def call_cleaners(data): #Call All cleaning functions

    filtered_texts=[]

    print('Cleaning Text Data..\n')

    for sent in tqdm(data):       

        s=text_cleaner_3(sent)

        filtered_texts.append(s)

    return filtered_texts



filtered_text=call_cleaners(train['comment_text'])
#Word Tokenization

print('Word Tokenization and Transforming them into Sequence..')

tokenizer=Tokenizer(num_words=30000)

tokenizer.fit_on_texts(filtered_text)

sequences=tokenizer.texts_to_sequences(filtered_text)

train_data_prepd=pad_sequences(sequences,maxlen=15)

word_index=tokenizer.word_index

print('Tokenization Done!')
embedding_matrix=np.zeros((30000,300))

print('Loading Embedding Matrix..\n')

for word,ix in tqdm(word_index.items()):

    if ix<50000:

        embed_vec=embedding_dict.get(word)

        if embed_vec is not None:

            embedding_matrix[ix]=embed_vec
#Lets see how many null embedding we have

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
print('Model Training')

model=Sequential()

model.add(Embedding(30000,300,input_length=15))

model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = True

model.add(Bidirectional(CuDNNLSTM(128,kernel_initializer='he_uniform',return_sequences=True)))

model.add(Bidirectional(CuDNNLSTM(196,kernel_initializer='he_uniform')))

model.add(Dropout(0.3))

model.add(Dense(1024,activation=None))

model.add(LeakyReLU())

model.add(Dense(2048,activation=None))

model.add(LeakyReLU())

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=Adam(0.001),metrics=['acc'])

model.summary()
#Call Backs

cb=callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)
result=model.fit(train_data_prepd,train_labels,epochs=20,batch_size=50,validation_split=0.2,callbacks=[cb])
#Plotting Model Accurcay and Loss

import matplotlib.pyplot as plt

acc = result.history['acc']

val_acc = result.history['val_acc']

loss = result.history['loss']

val_loss = result.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b',color='red', label='Training acc')

plt.plot(epochs, val_acc, 'b',color='blue', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', color='red', label='Training loss')

plt.plot(epochs, val_loss, 'b',color='blue', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()

#Test Predictions

test_data_filtered=call_cleaners(test['comment_text'])


#Tokenzie test set

print('Tokenizing Test Set..')

tokenizer.fit_on_texts(test_data_filtered)

sequences_test=tokenizer.texts_to_sequences(test_data_filtered)

test_data_prepd=pad_sequences(sequences_test,maxlen=15)

print('Done!')
#predictions

test_predictions=model.predict(test_data_prepd)
#Submission file

sub_file=pd.DataFrame({'id':test['id'],'prediction':test_predictions.reshape(-1)})

sub_file.to_csv('submission.csv',index=False)
#save model

model.save('comment_classifier.h5')