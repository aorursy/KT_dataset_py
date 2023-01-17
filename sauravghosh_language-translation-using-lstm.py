# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
from pickle import dump
from unicodedata import normalize
from pickle import load
from numpy.random import shuffle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_name = open(r"/kaggle/input/deu-english-datasets/deu.txt")
def load_doc(file_name):
    file = open(file_name,mode ='rt',encoding = 'utf-8')
    text = file.read()
    file.close()
    return text
def to_pairs(doc):
    lines = doc.strip().split("\n")
    pairs = [line.split("\t") for line in lines]
    return pairs
def clean_pairs(lines) :
    cleaned = list()
    re_punc = re.compile('[%s]'%re.escape(string.punctuation))
    re_print = re.compile('[^%s]'% re.escape(string.printable))
    for pair in lines :
        clean_pair = list()
        for line in pair :
            line = normalize("NFD",line).encode("ascii","ignore")
            line = line.decode("UTF-8")
            line = line.split()
            line = [word.lower() for word in line]
            line = [re_punc.sub('',w) for w in line]
            line = [re_print.sub('',w)for w in line]
            line = [w for w in line if w.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return cleaned
def save_clean_data(sentences,file_name):
    dump(sentences,open(file_name,'wb'))
    print("Saved : %s" % filename)
# load dataset
filename = r"/kaggle/input/deu-english-datasets/deu.txt"
doc = load_doc(filename)
pairs = to_pairs(doc)
clean_pairs = clean_pairs(pairs)
save_clean_data(clean_pairs,"english_german.pkl")
clean_pairs[4000]
def load_clean_data(filename):
    return(load(open(filename,'rb')))
raw_dataset = load_clean_data(r'/kaggle/input/english-german-clean-data/english_german.pkl')
n_sentences = 10000
dataset = raw_dataset[:n_sentences]
shuffle(dataset)
train, test = dataset[:9000], dataset[9000:]
save_clean_data(dataset,'english_german_both_subset.pkl')
save_clean_data(train,'english_german_train.pkl')
save_clean_data(test,'english_german_test.pkl')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
def load_clean_sentences(filename):
    return load(open(filename,"rb"))
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
def max_length(lines):
    return max(len(line.split()) for line in lines )
def encode_sequences(tokenizer,length,lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen = length, padding = 'post')
    return X
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence,num_classes = vocab_size)
        ylist.append(encoded)
    ylist = np.array(ylist)
    ylist = ylist.reshape(sequences.shape[0],sequences.shape[1],vocab_size)
    return ylist 
dataset = np.array(load_clean_sentences(r"/kaggle/input/language-modelling-data/english_german_both_subset.pkl"))
train = np.array(load_clean_sentences(r"/kaggle/input/language-modelling-data/english_german_train.pkl"))
test = np.array(load_clean_sentences(r"/kaggle/input/language-modelling-data/english_german_test.pkl"))
eng_tokenizer = create_tokenizer(dataset[:,0])
eng_vocab_size = len(eng_tokenizer.word_index) +1 
eng_length = max_length(dataset[:,0])
ger_tokenizer = create_tokenizer(dataset[:,1])
ger_vocab_size = len(ger_tokenizer.word_index)+1
ger_length = max_length(dataset[:,1])
trainX = encode_sequences(ger_tokenizer,ger_length,train[:,1])
trainY = encode_sequences(eng_tokenizer,eng_length,train[:,0])
trainY = encode_output(trainY, eng_vocab_size)
testX = encode_sequences(ger_tokenizer,ger_length,test[:,1])
testY = encode_sequences(eng_tokenizer,eng_length,test[:,0])
testY = encode_output(testY,eng_vocab_size)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(Dropout(.5))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(Dropout(.5))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 150)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(trainX, trainY, epochs=40, batch_size=100, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
import matplotlib.pyplot as plt
# summarize history for loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
from pickle import load
from keras.models import load_model
#model = load_model(r"/kaggle/input/language-model/model.h5")
model = load_model(r"/kaggle/working/model.h5")

from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu
def word_for_id(integer,tokenizer):
    for word , index in tokenizer.word_index.items():
        if index == integer :
            return word
    return None
def predict_sequence(model,tokenizer,source):
    prediction = model.predict(source,verbose = 0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers :
        word = word_for_id(i, tokenizer)
        if word is None :
            break
        target.append(word)
    
    return " ".join(target)
def evaluate_model(model,sources,raw_dataset):
    actual , predicted  = list(), list()
    for i , source in enumerate(sources):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model,eng_tokenizer,source)
        raw_target , raw_src = raw_dataset[i]
        #print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append([raw_target.split()])
        predicted.append(translation.split())
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
print('train')
evaluate_model(model, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, testX, test)