import numpy as np
import pandas as pd
import nltk
import re
import time
import random
import matplotlib.pyplot as plt
import sklearn
from numpy import genfromtxt
from tqdm import tqdm


#nltk.download('book')
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.book import *
#from gensim.models import Word2Vec
from string import punctuation
from nltk.tokenize import word_tokenize
train_data = pd.read_csv("../input/train.csv")
del train_data['qid']


test_data = pd.read_csv("../input/test.csv")

share = sum(train_data['target'] == 0) / len(train_data['target'])

print("The share of non insult comments is", round(share,4) * 100, "%")
train_data
def decontracted(phrase):
    
    """
    function that takes as input the most used english phrases and expands them to the actual
    words
    
    Input: 
    phrase - Phrases like "won't" or "don't"

    Returns: 
    The same phrase expanded to "will not" and "do not" respectively
    """
    
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"won’t", "will not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    hrase = re.sub(r"don’t", "do not", phrase)
    phrase = re.sub(r"don't", "do not", phrase)
    phrase = re.sub(r"can\'t", "cannot", phrase)
    phrase = re.sub(r"can't", "cannot", phrase)
    phrase = re.sub(r"can’t", "cannot", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"n't", " not", phrase)
    phrase = re.sub(r"n’t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"'re", " are", phrase)
    phrase = re.sub(r"’re", "are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"’s", "is", phrase)
    phrase = re.sub(r"'s", "is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"'d", " would", phrase)
    phrase = re.sub(r"’d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"'ll", " will", phrase)
    phrase = re.sub(r"’ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"'t", " not", phrase)
    phrase = re.sub(r"’t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"'ve", " have", phrase)
    phrase = re.sub(r"’ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"'m", " am", phrase)
    phrase = re.sub(r"’m", " am", phrase)
    phrase = re.sub(r'\w*@\w*','', phrase)
    
    return phrase


def preproc(data,word):


    sen = []

    for i in range(len(data)):
        sen.append(re.split(' |\\\\n|\\\\|\n\n|\n|xc2|xa0|x80|xe2|!|"|\.(?!\d)|\?(?!\d)|-|,',data[word][i]))

    for i in range(len(data)):  
        sen[i] = [word.lower() for word in sen[i]]
        sen[i] = [decontracted(word) for word in sen[i]]
    
    punct = list(punctuation)
    punct.append('``')
    punct.append("''")
    punct.append('--')
    punct.append('...')
    punct.append('')
    punct.append(',')
    punct.append("'")

    sentences = []
    
    for i in range(len(sen)):
        sentences.append([word for word in sen[i] if word not in punct])

        
    data = [' '.join(i) for i in sentences]
    data = np.asarray(data)    
    
    
    #[data[i].split() for i in range(len(data))]
        
    return data  



def word_length(data):

    length = []

    for i in range(len(data)):
        length.append(len(data[i].split()))
    
    max_len = max(length)
    return max_len
phrase = decontracted("I don't like this movie, I won't watch it again, I wasn’t in the house")
print('The sentence is going to be:',phrase)
data = preproc(train_data,'question_text')
data_test = preproc(test_data,'question_text')

all_data = np.concatenate((data,data_test),axis = 0)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
tokenizer.word_index
target = train_data['target']#[index]
target = np.asarray(target)
  

a = np.zeros(len(data))

for i in range(0,len(data)):
    a[i] = len(data[i].split())

print(sum(a<=90) / len(data))    
    
data = data[a <= 90]
target = target[a <=90]

#####################################


df = pd.DataFrame({'Comment': data, 'y':target})

X = df['Comment']
y = df['y']


max_len2 = 90
print(max_len2)



df_test = pd.DataFrame({'Comment': data_test})

test_X = df_test['Comment']

    

del data, data_test, all_data
words_to_vec = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    words_to_vec[word] = coefs
f.close()


all_embedds = np.stack(words_to_vec.values())
embedd_mean = all_embedds.mean()
embedd_std = all_embedds.std()

word_index = tokenizer.word_index
col_shape = words_to_vec['one'].shape[0]

embedd_matrix = np.random.normal(embedd_mean, embedd_std, (len(word_index),col_shape))

for word, i in word_index.items():
    if words_to_vec.get(word) is not None:
        embedd_matrix[i-1,:] = words_to_vec[word]
    
i = 0
word_to_index = {}


for word in word_index.keys():
    word_to_index[word] = i
    i = i + 1
#words_to_vec['is']
#word_to_index['is']
sum(embedd_matrix[2,:] - words_to_vec['what'])

del words_to_vec
def sentence_to_index(data,word_to_index,max_len,temp):
    
    """
    function that takes a sentences and gives the vector of indices back for all the words in the sentence
    
    Input: 
    data ... That is the data set, which contains the sentences to be translated to indices 
    word_to_index ... The dictionary that holds the index of any word in the word embedding 
    max_len... Maximum length of a sentence. If a sentence does not have maximum length, then the additional fields 
                are filled with zeros 
    temp ... This function is used twice
                1. Identify how many sentences have words which are not in the Glove6B embedding. If there are too
                   many unidentifable words, then this sentences will be taken out of the data (temp == 0)
                2. For the actual indexing of the vectors to find out the word vector of certain words.
                
    Output:
    Index_vector ... A matrix which returns the index of every word of a sentence in the corresponding word embedding
    """

    m = data.shape[0] # number of traing examples
    index_vector = np.zeros((m,max_len),dtype = 'int32') # Matrix of all sentence examples and corresponding indices
    
    for i in range(m):
        # Standardize all words in the sentence to lower case and split them 
        sentence_words = data[i].lower().split()
        
        j = 0
        
        for word in sentence_words:
            if word in word_to_index.keys():
                index_vector[i,j] = word_to_index[word]
            elif temp == 0:
                index_vector[i,j] = -1
            else:
                index_vector[i,j] = 0
            j = j + 1
              
    return index_vector
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02)
from keras.models import Model,Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, LeakyReLU,GRU,Flatten,MaxPooling1D,Bidirectional,GlobalMaxPooling1D,Conv1D,Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import regularizers
vocab_len = len(word_to_index) 
embedding_dim = 300
    
embedding_layer = Embedding(input_dim = vocab_len, output_dim = embedding_dim, weights = [embedd_matrix], trainable = False)
    
#input_shape = (max_len2,)


sentence_indices = Input(shape = (90,), dtype = 'int32')
    

# Propagate sentence_indices through your embedding layer, you get back the embeddings
embeddings = embedding_layer(sentence_indices)
X = Dropout(0.4)(embeddings)
# Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
# Be careful, the returned output should be a batch of sequences.
X = Bidirectional(GRU(units = 64, activation = 'tanh',return_sequences = True))(X)
# Add dropout with a probability of 0.7
X = Dropout(0.4)(X)
X = GlobalMaxPooling1D()(X)

X = Dense(units = 1)(X)
# Add a softmax activation
X = Activation('sigmoid')(X)
    
# Create Model instance which converts sentence_indices into X.
model = Model(inputs = sentence_indices, outputs = X)
X_train_index = sentence_to_index(np.asarray(X_train),word_to_index,max_len2,temp = 1)
Y_train_index = np.asarray(y_train)
model.summary()
opt = Adam(lr=0.001,decay = 10e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
class_weight = {0: 1.,
                1: 1.}
checkpoint = ModelCheckpoint('model1_check.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
temp1 = model.fit(X_train_index, Y_train_index, validation_split = 0.01,epochs = 2, batch_size = 1024,class_weight = class_weight, callbacks=[checkpoint])

X_test_index = sentence_to_index(np.asarray(X_test), word_to_index, max_len = max_len2, temp = 1)
#Y_test_index = np.eye(2)[np.asarray(y_test).reshape(-1)]
Y_test_index = np.asarray(y_test)
loss, acc = model.evaluate(X_test_index, Y_test_index)
print()
print("Test accuracy = ",acc * 100, "%")

y_pred = model.predict(sentence_to_index(np.asarray(X_test), word_to_index, max_len2, temp = 1))

#Finding the best value for F1 score threshold
F1score = {}

for n in np.arange(0.0, 0.51, 0.01):
    F1score[n] = sklearn.metrics.f1_score(y_test,y_pred > n)



import operator
threshold = max(F1score.items(), key=operator.itemgetter(1))[0]
test_predict = model.predict(sentence_to_index(np.asarray(test_X), word_to_index, max_len2, temp = 1))
prediction =  test_predict > threshold
pred = prediction * 1
test_data['prediction'] = pred
del test_data['question_text']
test_data
test_data.to_csv("submission.csv", index=False)
