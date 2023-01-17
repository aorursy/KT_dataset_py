import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import string

import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer,WordNetLemmatizer

from tqdm import tqdm

import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix

from keras.models import Sequential,load_model

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import pickle

from keras.optimizers import Adam

import tensorflow.compat.v1.keras.layers as kl

from keras.callbacks import ModelCheckpoint
train_df=pd.read_csv('../input/nlp-getting-started/train.csv')

test_df=pd.read_csv('../input/nlp-getting-started/test.csv')
train_df.head(10)
wl=WordNetLemmatizer()

ps=PorterStemmer()

sp=stopwords.words('english')

def clean_text(tweets):

    final_tmp=[]

    for tweet in tweets:

        #lower and remove punctuation

        tweet=tweet.translate(str.maketrans('','',string.punctuation)).lower()

        

        #Remove Hyperlinks

        tweet=re.sub(r'http\S+','',tweet)

        

        #Remove numbers and words containing numbers

        tweet=' '.join([i for i in tweet.split() if i.isalpha()])

        

        #Normalize words

        tweet=' '.join(wl.lemmatize(i,pos='a') for i in tweet.split())

        

        #Now stop words

        tweet=' '.join(i for i in tweet.split() if i not in sp)

        

        final_tmp.append(tweet)

    return final_tmp
cleaned_tweets=clean_text(train_df['text'])

print('Cleaned tweets :')

print(cleaned_tweets[:5])
X_train,X_val,y_train,y_val=train_test_split(cleaned_tweets,list(train_df['target']),test_size=0.1)
num_words=5000

maxlen=50

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)

X_val = tokenizer.texts_to_sequences(X_val)

X_train = pad_sequences(X_train, maxlen=maxlen)

X_val = pad_sequences(X_val, maxlen=maxlen)

word_index=tokenizer.word_index

embedding_path='../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

with open(embedding_path,'rb') as f:

    embedding_dict=pickle.load(f)

print('Found %s word vectors.' % len(embedding_dict))

embedding_matrix=np.zeros((num_words,300))

print('Loading Embedding Matrix..\n')

for word,ix in tqdm(word_index.items()):

    if ix<num_words:

        embed_vec=embedding_dict.get(word)

        if embed_vec is not None:

            embedding_matrix[ix]=embed_vec

        

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
model=Sequential()

model.add(layers.Embedding(num_words,300,input_length=maxlen))

model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = True

model.add(layers.GRU(16,return_sequences=True))

model.add(layers.GlobalMaxPooling1D())

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=Adam(0.0005),metrics=['acc'])

model.summary()
mc=ModelCheckpoint('classifier_0.h5',save_best_only=True,period=1,verbose=1)
history = model.fit(X_train, y_train,

                    epochs=10,

                    validation_data=(X_val, y_val),

                    batch_size=32,callbacks=[mc])
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

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
#Confusion matrix

model=load_model('classifier_0.h5')

y_preds=model.predict(X_train)

y_preds=[1 if i>0.5 else 0 for i in y_preds]

cm=confusion_matrix(y_train,y_preds)

sns.heatmap(cm,annot=True)
#test data

cleaned_test=clean_text(test_df['text'])

cleaned_test=tokenizer.texts_to_sequences(cleaned_test)

cleaned_test=pad_sequences(cleaned_test,maxlen=maxlen)

test_predictions=model.predict(cleaned_test)

test_predictions=[1 if i>0.5 else 0 for i in test_predictions]

sub=pd.DataFrame({'id':test_df['id'],'target':test_predictions})

sub.to_csv('submission.csv',index=False)