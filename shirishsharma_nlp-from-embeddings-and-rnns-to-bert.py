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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import string

from tqdm import tqdm



from gensim.parsing.preprocessing import remove_stopwords

from bs4 import BeautifulSoup

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from collections import OrderedDict

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.metrics import classification_report,f1_score



import tensorflow as tf

import tensorflow_hub as hub

from tensorflow import keras 

from keras import backend as K

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam



import torch

import transformers
train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')
print('Number of datapoints in the train dataset : ',train.shape[0])

print('Number of datapoints in the test dataset : ',test.shape[0])
train.head()
train.info()
test.info()
train.describe()
#removing any shortforms if present

def remove_shortforms(phrase):

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase



def remove_special_char(text):

    text = re.sub('[^A-Za-z0-9]+'," ",text)

    return text



def remove_wordswithnum(text):

    text = re.sub("\S*\d\S*", "", text).strip()

    return text



def lowercase(text):

    text = text.lower()

    return text



def remove_stop_words(text):

    text = remove_stopwords(text)

    return text



st = SnowballStemmer(language='english')

def stemming(text):

    r= []

    for word in text :

        a = st.stem(word)

        r.append(a)

    return r



def listToString(s):  

    str1 = " "   

    return (str1.join(s))



def remove_punctuations(text):

    text = re.sub(r'[^\w\s]','',text)

    return text



def remove_links(text):

    text = re.sub(r'http\S+', '', text)

    return text



lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    text = lemmatizer.lemmatize(text)

    return text



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)
Y = train['target']

train = train.drop('target',axis=1)

data = pd.concat([train,test],axis=0).reset_index(drop=True)

data.head()
for i in range(len(data['text'])):

    data['text'][i] = str(data['text'][i])
data['text'][1]
for i in range(len(data['text'])):

    data['text'][i] = remove_shortforms(data['text'][i])

    data['text'][i] = remove_special_char(data['text'][i])

    data['text'][i] = remove_wordswithnum(data['text'][i])

    data['text'][i] = lowercase(data['text'][i])

    data['text'][i] = remove_stop_words(data['text'][i])

    text = data['text'][i]

    text = text.split()

    data['text'][i] = stemming(text)

    s = data['text'][i]

    data['text'][i] = listToString(s)

    data['text'][i] = lemmatize_words(data['text'][i])
data['text'][1]
cv = CountVectorizer(ngram_range=(1,3))

text_bow = cv.fit_transform(data['text'])

print(text_bow.shape)
train_text = text_bow[:train.shape[0]] 

test_text = text_bow[train.shape[0]:] 
print(train_text.shape)

print(test_text.shape)
X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
# lr = LogisticRegression(max_iter=2000)



# params = {

#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],

#     'penalty': ['l1','l2']

# }



# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)

# clf.fit(X_train,Y_train)

# print(clf.best_params_)
lr = LogisticRegression(C=10,penalty='l2')

lr.fit(X_train,Y_train)

pred = lr.predict(X_test)

print("F1 score :",f1_score(Y_test,pred))

print("Classification Report \n\n:",classification_report(Y_test,pred))
lr = LogisticRegression(C=10,penalty='l2',max_iter=2000)

lr.fit(train_text,Y)

pred = lr.predict(test_text)

submit = pd.DataFrame(test['id'],columns=['id'])

print(len(pred))

submit.head()
submit['target'] = pred

submit.to_csv("realnlp.csv",index=False)
tfidf = TfidfVectorizer(ngram_range=(1,3))

text_tfidf = tfidf.fit_transform(data['text'])

print(text_tfidf.shape)
train_text = text_tfidf[:train.shape[0]] 

test_text = text_tfidf[train.shape[0]:] 

print(train_text.shape)

print(test_text.shape)
X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
# lr = LogisticRegression(max_iter=2000)



# params = {

#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],

#     'penalty': ['l1','l2']

# }



# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)

# clf.fit(X_train,Y_train)

# print(clf.best_params_)
lr = LogisticRegression(C=100,penalty='l2',max_iter=2000)

lr.fit(X_train,Y_train)

pred = lr.predict(X_test)

print("F1 score :",f1_score(Y_test,pred))

print("Classification Report :",classification_report(Y_test,pred))
print("Number of null values in data keywords column : ",data['keyword'].isnull().sum())
data.head()
data['keyword'] = data['keyword'].fillna("unknown")

data.head()
combined_text = [None] * len(data['text'])

for i in range(len(data['text'])):

    if data['keyword'][i] == 'unknown':

        combined_text[i] = data['text'][i]

    else:

        combined_text[i] = data['text'][i] + " " + data['keyword'][i] + " " + data['keyword'][i] + " " + data['keyword'][i]

data['combined_text'] = combined_text
data['combined_text'][88]
for i in range(len(data['combined_text'])):

    data['combined_text'][i] = str(data['combined_text'][i])
for i in range(len(data['combined_text'])):

    data['combined_text'][i] = remove_shortforms(data['combined_text'][i])

    data['combined_text'][i] = remove_special_char(data['combined_text'][i])

    data['combined_text'][i] = remove_wordswithnum(data['combined_text'][i])

    data['combined_text'][i] = lowercase(data['combined_text'][i])

    data['combined_text'][i] = remove_stop_words(data['combined_text'][i])

    text = data['combined_text'][i]

    text = text.split()

    data['combined_text'][i] = stemming(text)

    s = data['combined_text'][i]

    data['combined_text'][i] = listToString(s)

    data['combined_text'][i] = lemmatize_words(data['combined_text'][i])
data['combined_text'][88]
cv = CountVectorizer(ngram_range=(1,3))

text_bow = cv.fit_transform(data['combined_text'])

print(text_bow.shape)
train_text = text_bow[:train.shape[0]] 

test_text = text_bow[train.shape[0]:] 
X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
# lr = LogisticRegression(max_iter=2000)



# params = {

#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],

#     'penalty': ['l1','l2']

# }



# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)

# clf.fit(X_train,Y_train)

# print(clf.best_params_)
lr = LogisticRegression(C=1,penalty='l2',max_iter=2000)

lr.fit(X_train,Y_train)

pred = lr.predict(X_test)

print("F1 score :",f1_score(Y_test,pred))

print("Classification Report :",classification_report(Y_test,pred))
tfidf = TfidfVectorizer(ngram_range=(1,3))

text_tfidf = tfidf.fit_transform(data['combined_text'])

print(text_tfidf.shape)
train_text = text_tfidf[:train.shape[0]] 

test_text = text_tfidf[train.shape[0]:] 
X_train,X_test,Y_train,Y_test = train_test_split(train_text,Y,test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
# lr = LogisticRegression(max_iter=2000)



# params = {

#     'C' :[0.0001,0.001,0.01,0.1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,1,2,3,4,5,6,7,10,100,1000],

#     'penalty': ['l1','l2']

# }



# clf = RandomizedSearchCV(lr,params,n_jobs=-1,cv=10)

# clf.fit(X_train,Y_train)

# print(clf.best_params_)
lr = LogisticRegression(C=2,penalty='l2',max_iter=2000)

lr.fit(X_train,Y_train)

pred = lr.predict(X_test)

print("F1 score :",f1_score(Y_test,pred))

print("Classification Report :",classification_report(Y_test,pred))
print('Loading word vectors...')

word2vec = {}

with open(os.path.join('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'), encoding = "utf-8") as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

    for line in f:

        values = line.split() #split at space

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32') #numpy.asarray()function is used when we want to convert input to an array.

        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')
Y = train['target']

train = train.drop('target',axis=1)

data = pd.concat([train,test],axis=0).reset_index(drop=True)

text_data = data['text']
text_data
tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_data)

sequences = tokenizer.texts_to_sequences(text_data)
word2index = tokenizer.word_index

print("Number of unique tokens : ",len(word2index))
data_padded = pad_sequences(sequences,100)

print(data_padded.shape)
data_padded[6]
train_pad = data_padded[:train.shape[0]]

test_pad = data_padded[train.shape[0]:]
embedding_matrix = np.zeros((len(word2index)+1,200))



embedding_vec=[]

for word, i in tqdm(word2index.items()):

    embedding_vec = word2vec.get(word)

    if embedding_vec is not None:

        embedding_matrix[i] = embedding_vec
print(embedding_matrix[1])
model1 = keras.models.Sequential([

    keras.layers.Embedding(len(word2index)+1,200,weights=[embedding_matrix],input_length=100,trainable=False),

    keras.layers.LSTM(100,return_sequences=True),

    keras.layers.LSTM(200),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(1,activation='sigmoid')

])
model1.summary()
model1.compile(

  loss='binary_crossentropy',

  optimizer='adam',

  metrics=['accuracy'],

)
history1 = model1.fit(train_pad,Y,

                    batch_size=64,

                    epochs=10,

                    validation_split=0.2

)
plt.figure(figsize=(20,8))

plt.plot(history1.history['loss'], label='train')

plt.plot(history1.history['val_loss'], label='test')

plt.legend()

plt.grid()

plt.show()
plt.figure(figsize=(20,8))

plt.plot(history1.history['accuracy'], label='train')

plt.plot(history1.history['val_accuracy'], label='test')

plt.legend()

plt.grid()

plt.show()
model2 = keras.models.Sequential([

    keras.layers.Embedding(len(word2index)+1,200,weights=[embedding_matrix],input_length=100,trainable=False),

    keras.layers.GRU(100,return_sequences=True),

    keras.layers.GRU(200),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(1,activation='sigmoid')

])
model2.summary()
model2.compile(

  loss='binary_crossentropy',

  optimizer='adam',

  metrics=['accuracy'],

)
history2 = model2.fit(train_pad,Y,

                    batch_size=64,

                    epochs=10,

                    validation_split=0.2

)
plt.figure(figsize=(20,8))

plt.plot(history2.history['loss'], label='train')

plt.plot(history2.history['val_loss'], label='test')

plt.legend()

plt.grid()

plt.show()
plt.figure(figsize=(20,8))

plt.plot(history2.history['accuracy'], label='train')

plt.plot(history2.history['val_accuracy'], label='test')

plt.legend()

plt.grid()

plt.show()
model3 = keras.models.Sequential([

    keras.layers.Embedding(len(word2index)+1,200,weights=[embedding_matrix],input_length=100,trainable=False),

    keras.layers.Bidirectional(keras.layers.LSTM(100,return_sequences=True)),

    keras.layers.Bidirectional(keras.layers.LSTM(200)),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(1,activation='sigmoid')

])
model3.summary()
model3.compile(

  loss='binary_crossentropy',

  optimizer='adam',

  metrics=['accuracy'],

)
history3 = model3.fit(train_pad,Y,

                    batch_size=64,

                    epochs=10,

                    validation_split=0.2

)
plt.figure(figsize=(20,8))

plt.plot(history3.history['loss'], label='train')

plt.plot(history3.history['val_loss'], label='test')

plt.legend()

plt.grid()

plt.show()
plt.figure(figsize=(20,8))

plt.plot(history3.history['accuracy'], label='train')

plt.plot(history3.history['val_accuracy'], label='test')

plt.legend()

plt.grid()

plt.show()
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=3)
history = model3.fit(train_pad,Y,

                    batch_size=64,

                    epochs=30,

                    validation_split=0.2,

                    callbacks=[es]

)
plt.figure(figsize=(20,8))

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.grid()

plt.show()
plt.figure(figsize=(20,8))

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.grid()

plt.show()
submit = pd.DataFrame(test['id'],columns=['id'])

predictions = model3.predict(test_pad)

submit['target_prob'] = predictions

submit.head()
target = [None]*len(submit)

for i in range(len(submit)):

    target[i] = np.round(submit['target_prob'][i]).astype(int)

submit['target'] = target

submit.head()
submit = submit.drop('target_prob',axis=1)

submit.to_csv('real-nlp_lstm.csv',index=False)
train = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')
train.head()
Y = train['target']

train = train.drop('target',axis=1)

text_data_train = train['text']

text_data_test = test['text']
Y.value_counts()
tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

bert_model = transformers.TFBertModel.from_pretrained('bert-large-uncased')
def bert_encode(data,maximum_length) :

    input_ids = []

    attention_masks = []

  



    for i in range(len(data)):

        encoded = tokenizer.encode_plus(

        

          data[i],

          add_special_tokens=True,

          max_length=maximum_length,

          pad_to_max_length=True,

        

          return_attention_mask=True,

        

        )

      

        input_ids.append(encoded['input_ids'])

        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids),np.array(attention_masks)
train_input_ids,train_attention_masks = bert_encode(text_data_train,100)

test_input_ids,test_attention_masks = bert_encode(text_data_test,100)
train_input_ids[1]
train_attention_masks[1]
def create_model(bert_model):

    input_ids = tf.keras.Input(shape=(100,),dtype='int32')

    attention_masks = tf.keras.Input(shape=(100,),dtype='int32')

  

    output = bert_model([input_ids,attention_masks])

    output = output[1]

    output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)

    model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])

    return model
model = create_model(bert_model)

model.summary()
history = model.fit([train_input_ids,train_attention_masks],Y,

                    validation_split=0.2,

                    epochs=3,

                    batch_size=5)
# es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=3)
# history = model.fit([train_input_ids,train_attention_masks],Y,

#                     batch_size=10,

#                     epochs=10,

#                     validation_split=0.2,

#                     callbacks=[es]

# )
plt.figure(figsize=(20,8))

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.grid()

plt.show()
plt.figure(figsize=(20,8))

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.grid()

plt.show()
result = model.predict([test_input_ids,test_attention_masks])

result = np.round(result).astype(int)

submit = pd.DataFrame(test['id'],columns=['id'])

submit['target'] = result

submit.head()
submit.to_csv('real_nlp_bert.csv',index=False)