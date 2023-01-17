# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import bz2
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
trainfile = bz2.BZ2File('../input/train.ft.txt.bz2','r')
lines = trainfile.readlines()
lines[1]
docSentimentList=[]
def getSentimentList(docs,splitStr='__label__'):
    for i in range(len(docs)):
        #print('Processing doc ',i,' of ',len(docs))
        text=str(lines[i])
        #print(text)
        splitText=text.split(splitStr)
        secHalf=splitText[1]
        text=secHalf[2:len(secHalf)-1]
        sentiment=secHalf[0]
        #print('First half:',secHalf[0],'\nsecond half:',secHalf[2:len(secHalf)-1])
        docSentimentList.append([text,sentiment])
    print('It is Done')
    return docSentimentList
docSentimentList=getDocumentSentimentList(lines[:1000000],splitStr='__label__')
train_df = pd.DataFrame(docSentimentList,columns=['Text','Sentiment'])
train_df.head()
train_df['Sentiment'][train_df['Sentiment']=='1'] = 0
train_df['Sentiment'][train_df['Sentiment']=='2'] = 1
train_df.head(10)
train_df['Sentiment'].value_counts()
train_df['word_count'] = train_df['Text'].str.lower().str.split().apply(len)
train_df.head()
import string 
def remove_punc(s):
    table = str.maketrans({key: None for key in string.punctuation})
    return s.translate(table)
train_df['Text'] = train_df['Text'].apply(remove_punc)
train_df.shape
train_df.head()
len(train_df['word_count'][train_df['word_count']<=25])
train_df1 = train_df[:][train_df['word_count']<=25]
train_df1.head()
train_df1.head()
train_df1['Sentiment'].value_counts()
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
st_wd = text.ENGLISH_STOP_WORDS
c_vector = CountVectorizer(stop_words = st_wd,min_df=.0001,lowercase=1)
c_vector.fit(train_df1['Text'].values)
word_list = list(c_vector.vocabulary_.keys())
stop_words = list(c_vector.stop_words) 
len(stop_words),len(word_list)
def remove_words(raw_sen,stop_words):
    sen = [w for w in raw_sen if w not in stop_words]
    return sen
def reviewEdit(raw_sen_list,stop_words):
    sen_list = []
    for i in range(len(raw_sen_list)):
        raw_sen = raw_sen_list[i].split()
        sen_list.append(remove_words(raw_sen,stop_words))
    return sen_list
sen_list = reviewEdit(list(train_df1['Text']),stop_words)
from gensim.models import word2vec
wv_model = word2vec.Word2Vec(sen_list,size=100)
wv_model.wv.syn0.shape
wv_model.wv.most_similar("car")
def fun(sen_list,wv_model):
    word_set = set(wv_model.wv.index2word)
    X = np.zeros([len(sen_list),25,100])
    c = 0
    for sen in sen_list:
        nw=24
        for w in list(reversed(sen)):
            if w in word_set:
                X[c,nw] = wv_model[w]
                nw=nw-1
        c=c+1
    return X
X = fun(sen_list,wv_model)
from sklearn.model_selection import train_test_split
y = train_df1['Sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train.shape
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,LSTM, SimpleRNN ,GRU , Bidirectional,Input ,Concatenate, Multiply,Lambda,Reshape
input_st  = Input(shape=(25,100))
lstm1 = Bidirectional(GRU(200,input_shape=(25,100),activation='relu',return_sequences=True),merge_mode='mul')(input_st)
lstm2 = Bidirectional(GRU(1,input_shape=(25,100),activation='relu',return_sequences=True),merge_mode='mul')(lstm1)
print(lstm1.shape,' ',lstm2.shape)
lstm2 = Reshape((-1,))(lstm2)
lstm2 = Activation('sigmoid')(lstm2)
lstm2 = Reshape((-1,1))(lstm2)
mult = Multiply()([lstm1,lstm2])

add = Lambda(lambda x: K.sum(x,axis=1))(mult)
dense = Dense(100,activation='relu')(add)
output = Dense(1,activation='sigmoid')(dense)

model = Model(inputs=input_st, outputs=output)
print(model.summary())
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_train,y_train,validation_split=0.1,
          epochs=10, batch_size=512)
model.evaluate(X_test, y_test, batch_size=64)
prob_test = model.predict(X_test).reshape((-1,))
pred_test = np.array([1 if y>0.5 else 0 for y in prob_test])
y_test = y_test.astype('int')
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_test))
model.evaluate(X_train, y_train, batch_size=1024)
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
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
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,LSTM, SimpleRNN ,GRU , Bidirectional,Input ,Concatenate, Multiply,Lambda,Reshape,Conv2D,Conv1D
input_st  = Input(shape=(25,100))
lstm1 = Bidirectional(GRU(200,input_shape=(25,100),activation='relu',return_sequences=True),merge_mode='mul')(input_st)
lstm2 = Reshape((25,200,1))(lstm1)
atten = Conv2D(1,kernel_size=(25,1),activation='relu',use_bias=True)(lstm2)
##atten = Attention(25)(lstm1)
print(lstm1.shape,' ',atten.shape)
atten = Reshape((-1,))(atten)

dense = Dense(100,activation='relu')(atten)
output = Dense(1,activation='sigmoid')(dense)

model = Model(inputs=input_st, outputs=output)
print(model.summary())
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_train,y_train,validation_split=0.1,
          epochs=10, batch_size=512)
model.evaluate(X_test, y_test, batch_size=64)
model.evaluate(X_train, y_train, batch_size=1024)
prob_test = model.predict(X_test).reshape((-1,))
pred_test = np.array([1 if y>0.5 else 0 for y in prob_test])
y_test = y_test.astype('int')
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_test))
pred_test.shape,y_test.shape
!pip install --upgrade pip
!pip install nlpia
from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data
word_vectors = get_data("word2vec", limit=200)
def tokenize_and_vectorize(dataset):

	tokenizer = TreebankWordTokenizer() 
	vectorized_data =[]
	expected = []
	for sample in dataset:
		tokens = tokenizer.tokenize(sample[1])
		sample_vecs = []
		for token in tokens:
			try:
				sample_vecs.append(word_vectors[token])
			except KeyError:   
				pass

		vectorized_data.append(sample_vecs)

	return vectorized_data 

def pad_trunc(data,max_len):
	new_data=[]
	zero_vector=[]
	for _ in range(len(data[0][0])):
		zero_vector.append(0.0)

	for sample in data:  

		if len(sample) > max_len:
			temp = sample[:max_len]
		elif len(sample) < max_len:
			temp = sample
			additional_elems = max_len - len(sample)
			for _ in range(additional_elems):
				temp.append(zero_vector)
		else:
			temp = sample
		new_data.append(temp)
	return new_data


sample_1="I hate that the dismal weather had me down for so long, when will it break! Ugh, when does happiness return? The sun is blinding and the puffy clouds are too thin. I can't wait for the weekend"

vec_list = tokenize_and_vectorize([(1, sample_1)])

test_vec_list = pad_trunc(vec_list, 40)   
test_vec = (np.reshape(test_vec_list, (len(test_vec_list), 40, embedding_dims))) 
results = model.predict(test_vec)
print(results)



