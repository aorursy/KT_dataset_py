import pandas as pd

import numpy as np

from bs4 import BeautifulSoup 

import  re

import os

from imblearn.over_sampling import SMOTE







#nltk

import nltk



#preprocessing

from nltk.corpus import stopwords  #stopwords

from nltk import word_tokenize,sent_tokenize # tokenizing

from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet



# for part-of-speech tagging

from nltk import pos_tag



# for named entity recognition (NER)

from nltk import ne_chunk



# vectorizers for creating the document-term-matrix (DTM)

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer



# BeautifulSoup libraray

from bs4 import BeautifulSoup 

 # regex

from keras.preprocessing.text import Tokenizer

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.layers import Dense,Input,BatchNormalization,Dropout,Flatten

from keras.layers import Conv2D,MaxPooling2D

from keras.datasets import mnist

from keras.models import Sequential

from keras import backend as K
data=pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')

data.count()
text=data.iloc[:,[6,9]]

text.drop_duplicates(inplace=True)
def clean_review(text):

    review_text = re.sub("[^a-zA-Z]"," ",text)

    

    # 3. Converting to lower case and splitting

    word_tokens= review_text.lower().split()

    word_tokens= review_text.lower().split()

    

    # 4. Remove stopwords

    le=WordNetLemmatizer()

    stop_words= set(stopwords.words("english"))     

    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    

    cleaned_review=" ".join(word_tokens)

    return cleaned_review

text["cleaned_review"]=text.iloc[:,1].apply(clean_review)

text.count()
text = text.sample(frac=1).reset_index(drop=True)

print(text.shape)  # perfectly fine.

text_1=text

text_1.groupby('Score').count()

corpus=[]

for i in text_1.cleaned_review:

    corpus.append(i.split())

corpus
from gensim.models import Word2Vec

my_vec_1=Word2Vec(corpus, min_count=20,size=500 ,workers=10)



#my_vec_1.save("my_vec_500_all")
text_1.Score.unique()
text_1 = text_1.sample(frac=1).reset_index(drop=True)

tok = Tokenizer()

tok.fit_on_texts(text_1['cleaned_review'])

vocab_size = len(tok.word_index) + 1

encd_rev = tok.texts_to_sequences(text_1['cleaned_review'])

print(vocab_size)
maxi=-1

for i,rev in enumerate(text_1['cleaned_review']):

    tokens=rev.split()

    if(len(tokens)>maxi):

        maxi=len(tokens)

print(maxi)
max_rev_len=maxi  # max lenght of a review

vocab_size = len(tok.word_index) + 1  # total no of words

vocab_size
from keras.preprocessing.sequence import pad_sequences

pad_rev= pad_sequences(encd_rev, maxlen=max_rev_len)

pad_rev.shape

from sklearn.model_selection import train_test_split

y= keras.utils.to_categorical(text_1.Score)

x_train,x_val,y_train,y_val=train_test_split(pad_rev,y,test_size=0.30,random_state=42)

x_test,x_val,y_test,y_val=train_test_split(x_val,y_val,test_size=0.70,random_state=42)



print(x_test.shape)

print(x_val.shape)
embedding_matrix = np.zeros((vocab_size, 500))

for word, i in tok.word_index.items():

    if word in my_vec_1.wv.vocab:

        embedding_matrix[i] = my_vec_1.wv.word_vec(word)

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))



from keras.regularizers import L1L2

from keras.layers import ReLU,LSTM,GRU,CuDNNLSTM

from keras.layers import Dropout , Bidirectional

from keras.layers.embeddings import Embedding

model=Sequential()

model.add(Embedding(vocab_size,500,weights=[embedding_matrix], input_length=max_rev_len,trainable=False))



model.add(CuDNNLSTM(128 ,return_sequences=True))

model.add(Dropout(0.20))

model.add(CuDNNLSTM(128 ,return_sequences=False))

model.add(Dropout(0.20))



# model.add(CuDNNLSTM(64,return_sequences=False)) # loss stucks at about 

#model.add(Dense(256,activation='relu'))

model.add(Dense(6,activation='softmax'))



model.summary()

model.compile(optimizer=keras.optimizers.adam(),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=512,epochs=5,validation_data=(x_val,y_val))
history_1=history

yhat_probs = model.predict(x_test, verbose=0)

# predict crisp classes for test set

yhat_classes = model.predict_classes(x_test, verbose=0)

# reduce to 1d array

from sklearn.metrics import confusion_matrix

y_test_1=np.argmax(y_test , axis=1)

matrix = confusion_matrix(y_test_1, yhat_classes)

print(matrix)