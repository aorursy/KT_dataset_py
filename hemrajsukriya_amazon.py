# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import bz2

import pandas as pd

import re 

import numpy as np
#Import Dataset

train_file = bz2.BZ2File('../input/amazonreviews/train.ft.txt.bz2')

test_file = bz2.BZ2File('../input/amazonreviews/test.ft.txt.bz2')
#Reading Data set

train_file = train_file.readlines()

test_file = test_file.readlines()

print("Number of training reivews: " + str(len(train_file)))

print("Number of test reviews: " + str(len(test_file)))
#training on the first 10000 reviews in the  dataset

num_train = 100000

#Using 2000 reviews from test set

num_test = 20000#Using 200,000 reviews from test set



train_file = [x.decode('utf-8') for x in train_file[:num_train]]

test_file = [x.decode('utf-8') for x in test_file[:num_test]]
#Extracing Labels and Review from traing Dataset

train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]

train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]
#Extracing Labels and Review from test Dataset

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]

test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]
train = pd.DataFrame({'text':train_sentences,'label':train_labels})

test=pd.DataFrame({'text':test_sentences,'label':test_labels})
train.head()
train.describe()
train['number_of_words'] = train['text'].str.lower().str.split().apply(len)

train.head()
test['number_of_words'] = test['text'].str.lower().str.split().apply(len)

test.head()
import seaborn as sns

sns.set(style="darkgrid")

sns.countplot(x="label", data=train)
train['number_of_words'].plot(bins=50, kind='hist',figsize = (10,8)) 
train.hist(column='number_of_words', by='label',

           bins=50,figsize=(14,6))
import re



import nltk



def remove_url(text):

     url=re.compile(r"https?://\S+|www\.\S+")

     return url.sub(r" ",text)



def remove_html(text):

  cleanr = re.compile('<.*?>')

  return cleanr.sub(r" ",text)







def remove_num(texts):

   output = re.sub(r'\d+', '', texts)

   return output





import string

def remove_punc(text):

   table=str.maketrans(' ',' ',string.punctuation)

   return text.translate(table)







nltk.download('stopwords')

from nltk.corpus import stopwords

stop=set(stopwords.words("english"))

 

def remove_stopword(text):

   text=[word.lower() for word in text.split() if word.lower() not in stop]

   return " ".join(text)
train['text']=train.text.map(lambda x:remove_url(x))

train['text']=train.text.map(lambda x:remove_html(x))

train['text']=train.text.map(lambda x:remove_punc(x))

train['text']=train['text'].map(remove_num)

train['text']=train['text'].map(remove_stopword)
test['text']=test.text.map(lambda x:remove_url(x))

test['text']=test.text.map(lambda x:remove_html(x))

test['text']=test.text.map(lambda x:remove_punc(x))

test['text']=test['text'].map(remove_num)

test['text']=test['text'].map(remove_stopword)


import nltk



def Stemming(text):

   stem=[]

   from nltk.corpus import stopwords

   from nltk.stem import SnowballStemmer

  #is based on The Porter Stemming Algorithm

   stopword = stopwords.words('english')

   snowball_stemmer = SnowballStemmer('english')

   word_tokens = nltk.word_tokenize(text)

   stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]

   stem=' '.join(stemmed_word)

   return stem


train['text']=train['text'].map(Stemming)


test['text']=test['text'].map(Stemming)
import tensorflow as tf

max_length=100

vocab_size=12000

embedding_dim=64

trunc_type="post"

oov_tok="<OOV>"

padding_type="post"

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(train['text'])



word_index = tokenizer.word_index



training_sequences = tokenizer.texts_to_sequences(train['text'])

training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test['text'])

testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(word_index)
training_padded[1]
print(training_sequences[0])


from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM,GRU

from keras.preprocessing import sequence

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score    

from tensorflow.python.keras import models, layers, optimizers   

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D

from keras.layers.wrappers import Bidirectional

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(LSTM(256, dropout=0.2)))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.summary()





adam=Adam(lr=0.0001)


model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'] )


history=model.fit(training_padded,train['label'], epochs=15, batch_size=256,verbose = 1,callbacks = [EarlyStopping(monitor='val_accuracy', patience=2)],validation_data=(testing_padded,test['label']))
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

plot_graphs(history, "accuracy")

plot_graphs(history, "loss")
def Review(sentence):

   sequences = tokenizer.texts_to_sequences(sentence)

   padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



   prob=model.predict(padded)

   if prob>=0.8:

     print(5)

   elif prob>=0.6:

     print(4)

   elif prob>=0.4:

     print(3) 

   elif prob>=0.2:

     print(2)   

   else:

       print(1)
sentence=['Good Product + exactly in size']

Review(sentence)
sentence=['this is worst thing donot buy it']

Review(sentence)
# Predicting the Test set results

y_pred = model.predict(testing_padded)

y_pred = (y_pred > 0.5)

X_test=testing_padded

y_test=test['label']
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: %f' % accuracy)



precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)



recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)



f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)

 

# ROC AUC

auc = roc_auc_score(y_test, y_pred)

print('ROC AUC: %f' % auc)

# confusion matrix

matrix = confusion_matrix(y_test, y_pred)

print(matrix)
#Report

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
#Confusion Matrix

import seaborn as sns

sns.heatmap(matrix,annot=True,fmt='')
#ROC Curve



from sklearn.metrics import roc_curve

y_pred_keras = model.predict(X_test).ravel()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)



from sklearn.metrics import auc

auc_keras = auc(fpr_keras, tpr_keras)



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()

# Zoom in view of the upper left corner.

plt.figure(2)

plt.xlim(0, 0.2)

plt.ylim(0.8, 1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve (zoomed in at top left)')

plt.legend(loc='best')

plt.show()