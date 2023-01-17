import numpy as np
import pandas as pd 
from nltk.corpus import stopwords
import re
import string
from os import listdir
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import concatenate
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return(text)

def clean_doc(text,vocab):
    tokens = text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('',w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 1 and w.isalpha() and w in vocab]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return(tokens)

def clean_doc_2(text):
    tokens = text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('',w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 1 and w.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return(tokens)

def add_to_vocab(directory,vocab):
    for filename in listdir(directory):
        path = directory +"/"+filename 
        text = load_doc(path)
        tokens = clean_doc_2(text)
        vocab.update(tokens)
        
def save_file(lines,name):
    data = '\n'.join(lines)
    file = open(name,"w")
    file.write(data)
    file.close()
    
def process_docs(directory,vocab,is_train):
    documents=list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + "/" + filename
        text = load_doc(path)
        tokens = clean_doc(text,vocab)
        documents.append(tokens)
    return(documents)

def load_clean_dataset(vocab, is_train):
    neg = process_docs('../input/movie-review/movie_reviews/movie_reviews/neg', vocab, is_train)
    pos = process_docs('../input/movie-review/movie_reviews/movie_reviews/pos', vocab, is_train)
    docs = neg + pos
    labels = np.array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

def predict_sentiment(review, vocab, tokenizer, max_length, model):
    line = clean_doc(review, vocab)
    padded = encode_docs(tokenizer, max_length, [line])
    yhat = model.predict(padded, verbose=0)
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return percent_pos, 'NEGATIVE'
    return percent_pos, 'POSITIVE'

def predict_sentiment2(review, vocab, tokenizer, max_length, model):
    line = clean_doc(review, vocab)
    padded = encode_docs(tokenizer, max_length, [line])
    yhat = model2.predict([padded,padded,padded], verbose=0)
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return percent_pos, 'NEGATIVE'
    return percent_pos, 'POSITIVE'
# Create vocab bag

vocab = Counter()
add_to_vocab("../input/movie-review/movie_reviews/movie_reviews/neg",vocab)
add_to_vocab("../input/movie-review/movie_reviews/movie_reviews/pos",vocab)

vocab.most_common(50)
min_occurance = 2
tokens = [w for w,i in vocab.items() if i >=min_occurance]
save_file(tokens,'vocab.txt2')
# Load vocab
vocab_filename = 'vocab.txt2'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load training data
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
max_length = max([len(s) for s in train_docs])
vocab_size = len(tokenizer.word_index) + 1
Xtrain = encode_docs(tokenizer, max_length, train_docs)
Xtest = encode_docs(tokenizer, max_length, test_docs)
max_length
### 1 Layer Convolution model

# tf bug
Xtrain = np.matrix(Xtrain)
ytrain = np.array(ytrain)

# define model
model = Sequential()
model.add(Embedding(vocab_size,100,input_length=max_length))
model.add(Conv1D(filters = 32,kernel_size=8,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

# compile
model.compile(loss = 'binary_crossentropy',
             metrics = ['acc'],
             optimizer = 'adam')

#fit
#model.fit(Xtrain,ytrain,epochs = 10,verbose=2)
### Multichannel Convolution Model

## Define model
length = max_length
# channel 1
inputs1 = Input(shape=(length,))
embedding1 = Embedding(vocab_size, 100)(inputs1)
conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
inputs2 = Input(shape=(length,))
embedding2 = Embedding(vocab_size, 100)(inputs2)
conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
inputs3 = Input(shape=(length,))
embedding3 = Embedding(vocab_size, 100)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(dense1)
model2 = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

# compile
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model2.fit([Xtrain,Xtrain,Xtrain], ytrain, epochs=7, batch_size=16)

_, acc = model2.evaluate([Xtrain,Xtrain,Xtrain], ytrain, verbose=0)
_, acc = model2.evaluate([Xtest,Xtest,Xtest], ytest, verbose=0)
print('Train Accuracy: %f' % (acc*100))
print('Test Accuracy: %f' % (acc*100))
text = "a staple in cinematic history. beautiful from start to finish with engaging characters, a wonderful love story and some very intense visual effects. kate winslet and leonardo DIcaprio are truly the best things about the movie especially for 194 mins not since gone with the wind has there been such a memorable, tender, intimate, fond movie. superbly directed by james cameron, and academy award winning acting by everyone. the 1912 british passenger liner once again meets it's fate at the bottom of the atlantic ocean. also this movie has one of the all time best and memorable film scores ever made by james horner. "
a = predict_sentiment2(text, vocab, tokenizer, max_length, model)
print(a)