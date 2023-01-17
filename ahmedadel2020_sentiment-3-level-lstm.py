# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
LABEL_LENGTH = 2363
TRAIN_LENGTH = 1891
data = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
data = data[['airline_sentiment','text']]
data['airline_sentiment'].replace(['positive','negative','neutral'],[1,0,2],inplace=True)
neg_len = len(data[data['airline_sentiment'] == 0])
pos_len = len(data[data['airline_sentiment'] == 1])
neut_len = len(data[data['airline_sentiment'] == 2])

neg_train = data[data['airline_sentiment'] == 0][:TRAIN_LENGTH]
neg_test = data[data['airline_sentiment'] == 0][TRAIN_LENGTH:LABEL_LENGTH]

pos_train = data[data['airline_sentiment'] == 1][:TRAIN_LENGTH]
pos_test = data[data['airline_sentiment'] == 1][TRAIN_LENGTH:LABEL_LENGTH]

neut_train = data[data['airline_sentiment'] == 2][:TRAIN_LENGTH]
neut_test = data[data['airline_sentiment'] == 2][TRAIN_LENGTH:LABEL_LENGTH]

train_data = np.concatenate((neg_train,pos_train,neut_train),axis=0)
np.random.shuffle(train_data)
test_data = np.concatenate((neg_test,pos_test,neut_test),axis=0)
np.random.shuffle(test_data)
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import re
class Doc(object):
    def __init__(self):
        self.text = ''
        self.words = []
class Pipeline(object):
    def __init__(self,text=''):
        self.pipe = []
        self.doc = Doc()
    def add_pipe(self,fun):
        self.pipe.append(fun)
    def __call__(self,text):
        self.doc.text = text
        result = self.doc
        for fun in self.pipe:
            result = fun(result)
        return result
def tokenize_words(doc):
    doc.words = nltk.word_tokenize(doc.text.lower())
    return doc
def filter_words(doc):
    stopwords = [word for word in nltk.corpus.stopwords.words('english') if word not in ['out','on','off']]
    doc.words = list(filter(lambda word:re.match(r'[a-z]{2,}',word) and word not in stopwords,doc.words))
    return doc
def pos_tag(doc):
    doc.words = nltk.pos_tag(doc.words)
    return doc
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''
def word_lemmas(doc):
    for i in range(0,len(doc.words)):
        curr_word , tag = doc.words[i]
        new_word=''
        for letter in curr_word:
            if re.match('[a-z]',letter):
                 new_word+=letter
        position = get_wordnet_pos(tag)
        if position != '':
            new_word = nltk.stem.WordNetLemmatizer().lemmatize(new_word,pos=position)
        else:
            new_word=nltk.stem.WordNetLemmatizer().lemmatize(new_word)
        doc.words[i] = new_word
    return doc

import gensim

# Load pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin', binary=True)
def seq_to_vec(doc):
    result = np.zeros(300,dtype='float32')
    count = 0
    for word in doc.words:
        try:
            wv = model[str(word)]
            result+=np.nan_to_num(wv)
            count+=1
        except:
            continue
    if count > 0:
        result = result/count
    result = result.reshape((1,result.shape[0]))
    return result
pipe_line = Pipeline()
pipe_line.add_pipe(tokenize_words)
pipe_line.add_pipe(filter_words)
pipe_line.add_pipe(pos_tag)
pipe_line.add_pipe(word_lemmas)
pipe_line.add_pipe(seq_to_vec)
from keras.utils import to_categorical
train_y = train_data[:,0]
train_x = train_data[:,1]
test_x = test_data[:,1]
test_y =test_data[:,0]
train_y = train_y.astype('int32',copy=False)
test_y = test_y.astype('int32',copy=False)
train_y = to_categorical(train_y,num_classes = 3,dtype='int32')
test_y = to_categorical(test_y,num_classes = 3,dtype='int32')
train_y
temp_train_x = np.zeros((train_x.shape[0],1,300),dtype='float32')
for i in range(0,len(train_x)):
    temp_train_x[i] =  pipe_line(train_x[i])
temp_train_x
train_x = temp_train_x
temp_test_x = np.zeros((test_x.shape[0],1,300),dtype='float32')
for i in range(0,len(test_x)):
    temp_test_x[i] =  pipe_line(test_x[i])
test_x = temp_test_x
test_x
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model_lstm = Sequential()

# Recurrent layer
model_lstm.add(LSTM(units=64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1,use_bias=True, activation='relu'))

# Output layer
model_lstm.add(Dense(3, activation='softmax'))

# Compile the model
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model_lstm.fit(train_x,  train_y, 
                    batch_size=500, epochs=100)
accuracy = model_lstm.evaluate(test_x, test_y,verbose = 2) 
print('accuracy : ',accuracy[1])
model_lstm.save("model_sentiment.h5")
print("Saved model to disk")
from keras.models import load_model
model1 = load_model('model_sentiment.h5')
input_pred = pipe_line('i will buy this product').reshape((1,1,300))
np.argmax(model1.predict(input_pred).squeeze(0))
