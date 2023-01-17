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
#importing all the libraries

import sys , os , re , csv , codecs , numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,LSTM,Embedding,Dropout,Activation,BatchNormalization
from keras.layers import Bidirectional,GlobalMaxPool1D,GlobalAvgPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints , optimizers, layers
from keras.utils import to_categorical

#getting the train data
train = pd.read_csv('/kaggle/input/sentiment-analysis-of-tweets/train.txt')
#loading the test data
test = pd.read_csv('/kaggle/input/sentiment-analysis-of-tweets/test_samples.txt')
#diplay first 5 rows of train
train.head()
#one hot encoding the labels
df = pd.concat([train,pd.get_dummies(train['sentiment'])],axis=1)
#df.head()
train_data = df['tweet_text']
#train_data.head()
test_data = test['tweet_text']
test_data.head()
#creating the array of labels in serial with their respective texts
classes = ['neutral' , 'negative' , 'positive']
y = df[classes].values
y
#checking for null values in train and test data
train.isnull().any()
test.isnull().any()
#configuration  parameters
LATENT_DIM_DECODER = 400
BATCH_SIZE =128
EPOCHS = 20
LATENT_DIM = 400
NUM_SAMPLES = 10000
MAX_SEQUENCE_LEN = 1000
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 300

#NLTK python library for preprocessing
import nltk
#nltk.download('wordnet')
#for tokenization
from nltk.tokenize import RegexpTokenizer
#for stemming
from nltk.stem import WordNetLemmatizer,PorterStemmer
#for removing stopwords
from nltk.corpus import stopwords
#importing regex library of python
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
#function for performing all preproccing steps at once
def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens]#  if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

#make a dataframe of preprocessed text
df['cleanText']=train_data.map(lambda s:preprocess(s)) 
test['clean_text']=test['tweet_text'].map(lambda s:preprocess(s))
test_final = test['clean_text']
test_final
from keras.preprocessing.text import Tokenizer
#breaking the sentence into unique words/tokens
#expecting max tokens to be 20k
train_final = df['cleanText']
max_feat=40000
#tokenize sentence into list of words
tokenizer = Tokenizer(num_words=max_feat)#setting up tokenizer
#fiiting the tokenizer on out data
tokenizer.fit_on_texts(list(train_final))

train_final
tokenizer2 = Tokenizer(num_words=max_feat)#setting up tokenizer
#fiiting the tokenizer on out data
tokenizer2.fit_on_texts(list(test_final))
#converting text into sequence of numbers to feed in neural network
sequence_train = tokenizer.texts_to_sequences(train_final)
sequence_test = tokenizer2.texts_to_sequences(test_final)
# get the word to index mapping for input language
word2idx_inputs = tokenizer.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))

#LOADING PRETRAINED WORD VECTORS
# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open('/kaggle/glove.6B.300d.txt', encoding="utf8") as f:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

#EMBEDDING MATRIX
# prepare embedding matrix of words for embedding layer
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
    if(i < MAX_NUM_WORDS):
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector
max_len = [len(s) for s in sequence_train]
print(max(max_len))
from keras.preprocessing.sequence import pad_sequences
#scaling all the sequences to a fixed length
#dimension of input to the layer should be constant
#scaling each comment sequence to a fixed length to 200
#comments smaller than 200 will be padded with zeros to make their length as 200
max_len=1000
#pad the train and text sequence to be of fixed length (in keras input in lstm should be of fixed length sequnece)
x_train=pad_sequences(sequence_train,maxlen=max_len)
x_test=pad_sequences(sequence_test,maxlen=max_len)
from keras.layers import Embedding
# create embedding layer
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len,
  trainable=True
)
%matplotlib inline
from matplotlib import pyplot as plt
len_words = [len(words) for words in sequence_train]
#distribution of sequence
plt.hist(len_words, bins = np.arange(0,400,10))
plt.show()
# we can see that most of the comments have [0,50]  words
from keras.layers import Input
input = Input(shape=(max_len,))
#feeding the output of previous layer to the embedding layer that converts 
#the sequences into vector representation to detect relevance and context 
#of a particular word
embed_layer =embedding_layer(input)
import tensorflow as tf
from keras.layers.recurrent import LSTM
#passing the previous output as input to the BI_LSTM layer
LSTM_layer = tf.keras.layers.Bidirectional(LSTM(256, return_sequences=True, name='BI_lstm_layer'))(embed_layer)
sec_LSTM_layer = tf.keras.layers.Bidirectional(LSTM(256, return_sequences=True, name='BI2_lstm_layer'))(LSTM_layer)
batchnorm = BatchNormalization()(sec_LSTM_layer)
#dimension reduction using pooling layer
red_dim_layer = tf.keras.layers.GlobalAvgPool1D()(batchnorm)
##### adding dropout layer for better generalization
#setting value as 0.1 , which means 10$ of nodes will be randomly disabled
drop_layer = Dropout(0.55)(red_dim_layer)
#densely connected layer
dense1 = Dense(128,activation='elu')(drop_layer)
batchnorm2 = BatchNormalization()(dense1)
dense2 = Dense(128,activation='elu')(batchnorm2)
batchnorm3 = BatchNormalization()(dense2)
dense3 = Dense(128,activation='elu')(batchnorm3)
#adding another dropout layer
drop_layer2 = Dropout(0.55)(dense3)
#adding the output dense layer with sigmoid activation to get result 
#3  classes as output
output_dense = Dense(3,activation='softmax')(drop_layer2)
#connecting the inputs and outputs to create a model and compiling the model
from keras.optimizers import Adagrad,Adam,RMSprop
model = Model(inputs=input , outputs = output_dense)
model.compile(loss = 'categorical_crossentropy',
             optimizer = RMSprop(lr=0.001),
             metrics = ['accuracy'])
model.summary()
#Fitting the model 
batch_size=64
epochs = 30
model.fit(x_train,y,batch_size=batch_size,epochs = epochs,validation_split=0.2)