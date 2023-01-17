import numpy as np 

import pandas as pd 

import os



import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')



print(os.listdir("../input"))
import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from bs4 import BeautifulSoup

import re

from tqdm import tqdm

from keras.utils import to_categorical

import random

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.layers import Dense,Dropout,Embedding,LSTM

from keras.callbacks import EarlyStopping

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.models import Sequential
!unzip ../input/sentiment-analysis-on-movie-reviews/train.tsv.zip
!unzip ../input/sentiment-analysis-on-movie-reviews/test.tsv.zip
train= pd.read_csv("./train.tsv", sep="\t")

test = pd.read_csv("./test.tsv", sep="\t")



print(train.shape,test.shape)



train.head()
test.head()
train['Phrase'][0]
train.loc[train['SentenceId'] == 1]
dist = train.groupby(["Sentiment"]).size()



fig, ax = plt.subplots(figsize=(12,8))

sns.barplot(dist.keys(), dist.values);
#Function for cleaning the reviews, tokenize and lemmatize them.



def clean_sentences(df):

    reviews = []



    for sent in tqdm(df['Phrase']):

        

        #remove html content

        review_text = BeautifulSoup(sent).get_text()

        

        #remove non-alphabetic characters

        review_text = re.sub("[^a-zA-Z]"," ", review_text)

    

        #tokenize the sentences

        words = word_tokenize(review_text.lower())

    

        #lemmatize each word to its lemma

        lemma_words = [lemmatizer.lemmatize(i) for i in words]

    

        reviews.append(lemma_words)



    return(reviews)
#cleaned reviews for both train and test set retrieved



train_sentences = clean_sentences(train)

test_sentences = clean_sentences(test)



print(len(train_sentences))

print(len(test_sentences))
#Collect the dependent values and convert to one-hot encoded output using to_categorical



target=train.Sentiment.values

y_target=to_categorical(target)

num_classes=y_target.shape[1]

X_train,X_val,y_train,y_val = train_test_split(train_sentences,y_target,

                                             test_size=0.2,stratify=y_target)
#Geting the No. of unique words and max length of a review available in the list of cleaned reviews.



unique_words = set()

len_max = 0



for sent in tqdm(X_train):

    

    unique_words.update(sent)

    

    if(len_max<len(sent)):

        len_max = len(sent)

        

print(len(list(unique_words)))

print(len_max)
tokenizer = Tokenizer(num_words=len(list(unique_words)))

tokenizer.fit_on_texts(list(X_train))



X_train = tokenizer.texts_to_sequences(X_train)

X_val = tokenizer.texts_to_sequences(X_val)

X_test = tokenizer.texts_to_sequences(test_sentences)





X_train = sequence.pad_sequences(X_train, maxlen=len_max)

X_val = sequence.pad_sequences(X_val, maxlen=len_max)

X_test = sequence.pad_sequences(X_test, maxlen=len_max)



print(X_train.shape,X_val.shape,X_test.shape)
early_stopping = EarlyStopping(min_delta = 0.001, mode = 'max', monitor='val_acc', patience = 2)

callback = [early_stopping]
#Model using Keras LSTM



model=Sequential()



model.add(Embedding(len(list(unique_words)),300,input_length=len_max))

model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))

model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

model.add(Dense(100,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.005),metrics=['accuracy'])



model.summary()
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),

                  epochs=6, batch_size=256, verbose=1, callbacks=callback)
epoch_count = range(1, len(history.history['loss']) + 1)



plt.plot(epoch_count, history.history['loss'], 'r--')

plt.plot(epoch_count, history.history['val_loss'], 'b-')

plt.legend(['Training Loss', 'Validation Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.show()
y_pred=model.predict_classes(X_test)



sub_file = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv',sep=',')

sub_file.Sentiment=y_pred



sub_file.to_csv('Submission.csv',index=False)