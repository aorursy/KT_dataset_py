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
data_dir = '/kaggle/input/nlp-getting-started/'
train = pd.read_csv(data_dir+'train.csv')
train.head()
print(train.shape)
train['target'].value_counts()
train.isna().sum()
target = list(train['target'].values)
text = list(train['text'].values)
print(len(target),len(text))
text[:5],target[:5]
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 5000
EMBEDDING_DIM = 50
def preprocess_text(sents,max_vocab_size,max_seq_len,filters=None):
    if filters is not None:
        #with given filter
        tokenizer = Tokenizer(max_vocab_size,filters = filters)
    else:
        #with default filter
        tokenizer = Tokenizer(max_vocab_size)
    tokenizer.fit_on_texts(sents)
    processed_text = tokenizer.texts_to_sequences(sents)
    max_len = min(max_seq_len,max(len(line) for line in processed_text))
    padded_text = pad_sequences(processed_text,maxlen=max_len,padding ='post')
    return padded_text,tokenizer

def load_glove_vectors(file_path):
    print('loading glove embeddings')
    with open(file_path) as f:
        word2vec = {}
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:],dtype = 'float32')
            word2vec[word] = vec
        print('Found {} words'.format(len(word2vec)))
    return word2vec

def get_embedding_matrix(file_path,num_words,embedding_dim,word2idx,max_vocab_size):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    word2vec = load_glove_vectors(file_path)
    for word,i in word2idx.items():
        if i < max_vocab_size:
            emb_vec = word2vec.get(word)
            if emb_vec is not None:
                embedding_matrix[i] = emb_vec
    return embedding_matrix
processed_text,tokenizer = preprocess_text(text,MAX_VOCAB_SIZE,MAX_SEQUENCE_LENGTH)
print('sample text : ',text[0])
print('sample processed text : ',processed_text[0])
max_seq_len = len(processed_text[0])
print('No of sentences : {}'.format(len(processed_text)))
print('Max sequence length : {}'.format(max_seq_len))
word2idx = tokenizer.word_index
print('total number of words : {}'.format(len(word2idx)+1))
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
print('Number of words : {}'.format(num_words))
embed_matrix = get_embedding_matrix('glove.6B.50d.txt',num_words,EMBEDDING_DIM,word2idx,MAX_VOCAB_SIZE)
print(embed_matrix.shape)
## creating the model
from tensorflow.keras.layers import Input,Dense,LSTM,SimpleRNN,Embedding
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
LATENT_DIM = 50
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32
EPOCHS = 10
#train validation split
from sklearn.model_selection import train_test_split
train_x,valid_x,train_y,valid_y = train_test_split(processed_text,np.array(target),test_size = VALIDATION_SPLIT)
print('training shape, X : ',train_x.shape,' , Y : ',train_y.shape)
print('validation shape, X :',valid_x.shape,'Y : ',valid_y.shape)
embedding_layer = Embedding(num_words,EMBEDDING_DIM,weights = [embed_matrix])

input_ = Input(shape = (max_seq_len,))
X = embedding_layer(input_)
X = LSTM(LATENT_DIM)(X)
output = Dense(1,activation = 'sigmoid')(X)

model = Model(inputs = input_,outputs = output)
model.compile(loss='binary_crossentropy',optimizer=Adam(0.0005),metrics=['accuracy'])
model.summary()
print('Training model')
r = model.fit(train_x,train_y,batch_size = BATCH_SIZE,epochs = EPOCHS,validation_data = (valid_x,valid_y))
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(r.history['accuracy'],label = 'train acc')
plt.plot(r.history['val_accuracy'],label = 'valid acc')
plt.legend()
plt.plot(r.history['loss'],label = 'train loss')
plt.plot(r.history['val_loss'],label = 'valid loss')
plt.legend()
