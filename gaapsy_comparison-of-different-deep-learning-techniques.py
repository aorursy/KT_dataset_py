from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.datasets import reuters
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Masking
from keras.callbacks import EarlyStopping
from keras.layers import Convolution1D, GlobalMaxPooling1D
import collections
PATH = "../input/Tweets.csv"

data=pd.read_csv(PATH)
data= data.copy()[['airline_sentiment', 'text']]

max_words = 10000
batch_size = 32
nb_epoch = 10
maxlen = 12
max_features = 10000
nb_filter = 250
filter_length = 3
hidden_dims = 250
total=0
coall=collections.Counter()



data.sample(5)
Counter(data.airline_sentiment)
def review_to_words( review ):
    review_text = review
    no_hasthtags = re.sub("#\w+", " ", review_text)
    no_url = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", no_hasthtags )
    no_tag = re.sub("@\w+", " ", no_url)
    no_punctions = re.sub("[^a-zA-Z]", " ", no_tag) 
    wordslower= no_punctions.lower()
    words = word_tokenize(wordslower)  
    stopswd = set(stopwords.words("english"))                  
    meaningful_wd = [w for w in words if not w in stopswd]
    str=' '.join(meaningful_wd)   
    return(str) 
clean_text = []
for tweet in data['text']:
    clean= review_to_words(tweet)
    clean_text.append(clean)
for i in range(0,len(data['airline_sentiment'])): 
     if data['airline_sentiment'][i]=='negative':
        data['airline_sentiment'][i]=0
     if data['airline_sentiment'][i]=='positive':
        data['airline_sentiment'][i]=1
     if data['airline_sentiment'][i]=='neutral':
        data['airline_sentiment'][i]=2

earlystop = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001, verbose=1, mode='auto')
callbacks_list = [earlystop]
data['text'] = clean_text
total=0
history = []
for t in range (0,10):
     cv=10
     k = [int((len(data['text']))/cv*j) for j in range(cv+1)]
     X_test, y_test= data['text'][k[t]:k[t+1]], data['airline_sentiment'][k[t]:k[t+1]]
     X_train, y_train =pd.concat([data['text'][:k[t]],data['text'][k[t+1]:]]), pd.concat([data['airline_sentiment'][:k[t]],data        ['airline_sentiment']  [k[t+1]:]])
     nb_classes = 3
     train_data=[]
     for i in X_train:
        train_data.append(i)
     test_data=[]
     for i in X_test:
        test_data.append(i)
     tokenizer = Tokenizer(num_words=max_words)
     X_train = tokenizer.fit_on_texts(train_data)
     X_test = tokenizer.fit_on_texts(test_data)
     X_train = tokenizer.texts_to_matrix(train_data,mode='binary')
     X_test = tokenizer.texts_to_matrix(test_data,mode='binary')
     y_train = np_utils.to_categorical(y_train, nb_classes)
     y_test = np_utils.to_categorical(y_test, nb_classes)
     model = Sequential()
     model.add(Dense(10, input_shape=(max_words,)))
     model.add(Activation('relu'))
     model.add(Dropout(0.5))
     model.add(Dense(3))
     model.add(Activation('softmax'))
     model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
     history.append(model.fit(X_train, y_train,
                    epochs=nb_epoch, batch_size=batch_size,
                    callbacks = callbacks_list, validation_split=0.1))
     score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
     total=total+score[1]
     t=t+1


print(model.summary())
print(total)
accuracy=total/10
print(accuracy)


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
for x in history :
    plot_model_history(x)
def language_preprocessing(x_train,x_test,x_dev):
   each_critique=[]
   train=[]
   test=[]
   dev=[]
   t=Tokenizer()
   t.fit_on_texts(data['text'])
   dictionnary=t.word_index
   for element in x_train:
       words = word_tokenize(element)
       for element in words:
                each_critique.append(dictionnary[element])
       train.append(each_critique)
       each_critique=[]
   for element in x_test:
       words = word_tokenize(element)
       for element in words:
                each_critique.append(dictionnary[element])
       test.append(each_critique)
       each_critique=[]
   for element in x_dev:
       words = word_tokenize(element)
       for element in words:
                each_critique.append(dictionnary[element])
       dev.append(each_critique)
       each_critique=[]
   return(train,test,dev)

histoLSTM = []
for t in range (0,10):
     cv=10
     data['text'] = clean_text
     k = [int((len(data['text']))/cv*j) for j in range(cv+1)]
     X_test, y_test= data['text'][k[t]:k[t+1]], data['airline_sentiment'][k[t]:k[t+1]]
     X_train, y_train =pd.concat([data['text'][:k[t]],data['text'][k[t+1]:]]), pd.concat([data['airline_sentiment'][:k[t]],data        ['airline_sentiment']  [k[t+1]:]])
     X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
     y_train = np_utils.to_categorical(y_train, 3)
     y_test = np_utils.to_categorical(y_test, 3)
     y_dev = np_utils.to_categorical(y_dev, 3)
     train_data=[]
     for i in X_train:
       train_data.append(i)
     test_data=[]
     for i in X_test:
       test_data.append(i)
     dev_data=[]
     for i in X_dev:
       dev_data.append(i)
     X_train,X_test,X_dev=language_preprocessing(train_data,test_data,dev_data)
     X_train= sequence.pad_sequences(X_train, maxlen=maxlen)
     X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
     X_dev= sequence.pad_sequences(X_dev, maxlen=maxlen)
     model = Sequential() 
     model.add(Embedding(max_features, 50, mask_zero=True,input_length=12))
     model.add(LSTM(3, dropout=0.3, recurrent_dropout=0.05))
     model.add(Dense(3))
     model.add(Activation('softmax'))
     model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
     histoLSTM.append(model.fit(X_train, y_train, batch_size=batch_size, epochs=10, callbacks = callbacks_list,
         validation_data=(X_dev, y_dev)))
     score, acc = model.evaluate(X_test, y_test,
                           batch_size=batch_size)
     total=total+acc
     t=t+1


print("************************************")
print(model.summary()) 
print(total)
accuracy=total/10
print(accuracy)
for x in histoLSTM:
    plot_model_history(x)

histoCNN=[]
for t in range (0,10):
     cv=10
     data['text'] = clean_text
     k = [int((len(data['text']))/cv*j) for j in range(cv+1)]
     X_test, y_test= data['text'][k[t]:k[t+1]], data['airline_sentiment'][k[t]:k[t+1]]
     X_train, y_train =pd.concat([data['text'][:k[t]],data['text'][k[t+1]:]]), pd.concat([data['airline_sentiment'][:k[t]],data        ['airline_sentiment']  [k[t+1]:]])
     X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
     y_train = np_utils.to_categorical(y_train, 3)
     y_test = np_utils.to_categorical(y_test, 3)
     y_dev = np_utils.to_categorical(y_dev, 3)
     train_data=[]
     for i in X_train:
       train_data.append(i)
     test_data=[]
     for i in X_test:
       test_data.append(i)
     dev_data=[]
     for i in X_dev:
       dev_data.append(i)
     X_train,X_test,X_dev=language_preprocessing(train_data,test_data,dev_data)
     X_train= sequence.pad_sequences(X_train, maxlen=maxlen)
     X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
     X_dev= sequence.pad_sequences(X_dev, maxlen=maxlen)
     model = Sequential()
     embedding_layer = Embedding(max_features, 50, input_length=12)
     model.add(embedding_layer)
     model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
     model.add(GlobalMaxPooling1D())
     model.add(Dense(hidden_dims))
     model.add(Dropout(0.2))
     model.add(Activation('relu'))
     model.add(Dense(3))
     model.add(Activation('softmax'))
     model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
     histoCNN.append(model.fit(X_train, y_train,
         batch_size=batch_size, callbacks=callbacks_list, 
         nb_epoch=nb_epoch,
         validation_data=(X_dev, y_dev)))
     acc = model.evaluate(X_test, y_test,
                           batch_size=batch_size)
     print('Test accuracy:', acc[1])
     total=total+acc[1]
     t=t+1




print("************************************")
print(model.summary()) 
print(total)
accuracy=total/10
print(accuracy)
for x in histoCNN:
    plot_model_history(x)
