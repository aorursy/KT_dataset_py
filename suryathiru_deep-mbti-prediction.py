import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
from pprint import pprint

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/mbti-type/mbti_1.csv')
personalities = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}
data.head()
def replace_symbols(text):
    text = re.sub('\|\|\|', ' ', text)
    text = re.sub('https?\S+', '<URL>', text)
    return text

data['cleaned_posts'] = data['posts'].apply(replace_symbols)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight
import gensim
# covert the glove model to word2vec format
from gensim.scripts.glove2word2vec import glove2word2vec

glove_dir = '../input/glove-global-vectors-for-word-representation/'
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.word2vec'
glove2word2vec(glove_dir + glove_input_file, word2vec_output_file)
# load the vectors
from gensim.models import KeyedVectors
glovec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# TEST MODEL: calculate: (king - man) + woman = ?
result = glovec_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
glovec_weights = glovec_model.wv.syn0
vocab_size, embedding_size = glovec_weights.shape
# convert to glove index
def word2idx(word):
    idx = glovec_model.wv.vocab.get(word)
    if not idx:
        return None
    return idx.index

def idx2word(idx):
    return glovec_model.wv.index2word[idx]

def convert_text(doc):
    return [word2idx(word) for word in doc]

# data['encoded_posts'] = data['cleaned_posts'].apply(convert_text)
tok = Tokenizer()
tok.fit_on_texts(data['cleaned_posts'])
docs = tok.texts_to_sequences(data['cleaned_posts'])

MAX_LEN = 1000
padded = pad_sequences(docs, maxlen=MAX_LEN, padding='post')

vocab_size = len(tok.word_index) + 1 # vocab size from data
print(vocab_size)
# generate embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_size))
for word, i in tok.word_index.items():
    idx = word2idx(word)
    if idx is not None:
        emb_vec = glovec_weights[idx]        
        embedding_matrix[i] = emb_vec
        
embedding_matrix.shape
le = LabelEncoder()

X, Y = padded, le.fit_transform(data['type'])
X.shape, Y.shape
# handle imbalance in dataset
class_weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
class_weights = dict(enumerate(class_weights))
class_weights
def create_RNN():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, weights=[embedding_matrix], trainable=False, input_length=MAX_LEN))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='softmax'))
    model.compile(Adam(0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
rnn = create_RNN()
rnn.summary()
callbacks = [EarlyStopping(min_delta=0.001, verbose=1)]
model_info = rnn.fit(X, Y, validation_split=0.15, batch_size=64, epochs=5, callbacks=callbacks)#, class_weight=class_weights)
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    
plot_model_history(model_info)
import keras.backend as K

print(len(rnn.layers))
out = K.function([rnn.inputs[0], K.learning_phase()], [rnn.layers[1].output])
print(out([X[:2],0]), X[:2])
!rm glove.word2vec
!du -ahd 1
model_dir = 'mbit_rnn_model.json'
weights_dir = 'mbit_rnn_weights.h5'

model_json = rnn.to_json()
with open(model_dir, 'w') as file:
    file.write(model_json)

rnn.save_weights(weights_dir)
!du -ahd 1