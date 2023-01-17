def plot_model_output(history,epochs):
    plt.figure()
    plt.plot(range(epochs,),history.history['loss'],label = 'training_loss')
    plt.plot(range(epochs,),history.history['val_loss'],label = 'validation_loss')
    plt.legend()
    plt.figure()
    plt.plot(range(epochs,),history.history['acc'],label = 'training_accuracy')
    plt.plot(range(epochs,),history.history['val_acc'],label = 'validation_accuracy')
    plt.legend()
    plt.show()
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
data = pd.read_csv('/kaggle/input/imdb-review-dataset/imdb_master.csv',encoding='ISO-8859-1')
print(data.columns)
sns.countplot(x='label',data=data)
data = data[data.label!='unsup']
sns.countplot(x='label',data=data)
data['out'] = data['label'] 
data['out'][data.out=='neg']=0
data['out'][data.out=='pos']=1
# Another way data['out'] = data['out'].map({1:'pos',0:'neg'})
np.unique(data.out)
sns.countplot(y='out',data=data)
req_data = data[['review','out']]
req_data.head()
texts = np.array(req_data.review)
labels = np.array(req_data.out)
print(texts.shape, labels.shape)
# num_words: Top No. of words to be tokenized. Rest will be marked as unknown or ignored.
tokenizer = Tokenizer(num_words=20000) 
# tokenizing based on "texts". This step generates the word_index and map each word to an integer other than 0.
tokenizer.fit_on_texts(texts)

# generating sequence based on tokenizer's word_index. Each sentence will now be represented by combination of numericals
# Example: "Good movie" may be represented by [22, 37]
seq = tokenizer.texts_to_sequences(texts)

# padding each numerical representation of sentence to have fixed length.
padded_seq = np.array(pad_sequences(seq,maxlen=100))

#word_index of each token
word_index = tokenizer.word_index

# for shuffling
indices = np.arange(padded_seq.shape[0])
np.random.shuffle(indices)

# texts = texts[indices]
# labels = labels[indices]
# this is how word_index looks like. It's a dict where each word has its own unique key 
# with which the word is represented in sequence.

print(len(word_index))
print(padded_seq.shape)
print()
# GloVe embedding matrix. This is like a pre-trained model. In an embedding matrix, each word is represented by a dense vector.

vec_representations = {}
f = open('/kaggle/input/glove6b/glove.6B.100d.txt','r')

sample = True

for line in f:
    if sample:
        print("Sample weight of word: ")
        print(line)
        sample=False
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype = 'float32')
    vec_representations[word] = coefs
    
f.close()

print(f'Found {len(vec_representations)} words')
max_words = 20000
embedding_dim = 64
embedding_matrix = np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    vec_representation = vec_representations.get(word)
    if vec_representation is not None:
        embedding_matrix[i]=vec_representation
model = Sequential()
embedding = Embedding(max_words,embedding_dim,input_length = 100,name='embedding')
model.add(embedding)
#model.add(LSTM(32, return_sequences = True))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
# model.layers[0].set_weights=[(embedding_matrix)]
# model.layers[0].trainable=True
model.layers[0].get_weights()
np.asarray(labels).astype(np.uint8)
model.compile(optimizer='adagrad',loss='binary_crossentropy',metrics=['acc'])

# Change the epochs to 10 here.
history = model.fit(padded_seq,np.asarray(labels).astype(np.uint8),epochs=10,validation_split=0.3)
plot_model_output(history, 10)
import numpy as np
timesteps = 10000 # No. of timesteps in input_sequence
input_features = 32 # Dimensionality of input_feature space
output_features = 64 # Dimensionality of output_features space

inputs = np.random.random((timesteps,input_features)) #random data for example 10000 X 32

state_t = np.zeros((output_features,)) # (64,)

W = np.random.random((output_features,input_features)) # Weight matrix for current_state (64 X 32)
U = np.random.random((output_features,output_features)) #Weight matrix for previous state (64 X 64)

b = np.random.random((output_features,)) # bias to be added to output of each state (64,)

successive_outputs = [] # output of each timestep

for input_t in inputs:
    # input_t is a vector of shape (input_features,)
    output_t = np.tanh(np.dot(W,input_t)+np.dot(U,state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_outputs = np.concatenate(successive_outputs,axis=0)

final_outputs
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense, Flatten
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd
model = Sequential()
model.add(Embedding(10000,32)) # 10K is no of unique words and 32 is shape of each word's representation
model.add(SimpleRNN(units=32,return_sequences = True)) #units is dimensionality of output space. https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
model.add(SimpleRNN(units=32,return_sequences = True)) 
model.add(SimpleRNN(units=32,return_sequences = True)) 
seq = model.add(SimpleRNN(units=32,return_sequences = True)) 
model.summary()
max_features = 10000 # max num of words. others will be marked as unknown
maxlen = 500 # maximum length of each review
batch_size = 32
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
indices = list(range(len(x_train)))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]
model = Sequential()
model.add(Embedding(max_features,32)) # here 32 is no of bits with which a sentence with maxlen(500 set above) will be represented.
model.add(SimpleRNN(32)) # 32 is output space dimensions
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',metrics=['acc'])

history = model.fit(x_train,
          y_train,
         epochs=4,
         batch_size = batch_size,
         validation_split=0.2)
plt.figure()
plt.plot(range(4,), history.history['acc'],'g',label = 'training acc')
plt.plot(range(4,), history.history['val_acc'],'*',label = 'val_acc')
plt.title("Training and validation acc")
plt.legend()
plt.show()
plt.figure()
plt.plot(range(4,),history.history['loss'],label = 'training_loss')
plt.plot(range(4,),history.history['val_loss'],label = 'validation_loss')
plt.legend()
plt.show()

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense, Flatten, LSTM
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd
max_features = 20000 # max num of words. others will be marked as unknown
maxlen = 1000 # maximum length of each review
batch_size = 64
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
y_train
model = Sequential()
model.add(Embedding(max_features,64)) # here 64 is no of bits with which a sentence with maxlen(500 set above) will be represented.
model.add(LSTM(64,dropout=0.3))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(x_train,y_train,
         validation_split = 0.2,
         epochs = 1,
         batch_size=batch_size)

plot_model_output(history,1)
model.layers[-2].get_weights()
