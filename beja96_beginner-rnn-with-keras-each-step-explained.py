#Load General packages

import os

import numpy as np

import pandas as pd # for data processing

import matplotlib.pyplot as plt #for vizualization of data

from tqdm import tqdm #for progress information

tqdm.pandas()
#Load "train" dataset

data = pd.read_csv("../input/train.csv")



#Load "test" dataset

submission_data = pd.read_csv("../input/test.csv")
#Load "glove.840B.300d" data

embeddings_index = {}    #creates empty list

glove = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt') #opens the test document for reading

for line in tqdm(glove): #for every line in this text do the following        (tqdm: and show the progress)

    values = line.split(" ")  #splits the string every time there is a space into seperate strings

    word = values[0] #the first string in this text file is always the word

    coefs = np.asarray(values[1:], dtype='float32') # the following strings are the "explanation"

    embeddings_index[word] = coefs #the list is now filled with entries consisting of the word and the respective "explanations" (word vectors)

glove.close() #closes the file such that is not possible to read it anymore



print('The dictionary contains %s word vectors.' % len(embeddings_index))
#Example on how Glove represents the word "kitchen"

word = "kitchen"

print("The vector of", word, "in the dictionary is", embeddings_index[word])

data.head() # shows the first 5 rows of a dataset
data.info() #presents general information to the dataset
pd.options.display.max_colwidth = 300 # for setting the width of the table longer



#Sincere questions

data.loc[data['target'] == 0].head(10)
#Insincere questions

data.loc[data['target'] == 1].head(10)
fig1, ax1 = plt.subplots()

ax1.pie(data["target"].value_counts(), explode=(0, 0.3), labels= ["Sincere", "Insincere"], autopct='%1.1f%%',

        shadow=True, startangle=45)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
from sklearn.model_selection import train_test_split



train, test = train_test_split(data, #performing the split

                 test_size = 0.3)

train = train.reset_index(drop=True) #thus the df counts from 0 to x and does not spring from one number to another number

test = test.reset_index(drop = True)

print ("The training dataset has the shape:" , train.shape)

print ("The test dataset has the shape:", test.shape)
X_train = train.iloc[:,1] #Takes all rows of the first column as new dataset

Y_train = np.array(train.iloc[:, 2]) #Takes all rows of the second column as new dataset

X_test = test.iloc[:,1]

Y_test = np.array(test.iloc[:, 2])



print(X_train.shape)

print(Y_train.shape)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences 

from keras.preprocessing import sequence 



#features

tokenizer = Tokenizer(filters='', lower=False) #To ensure that no preprocessing is done at all

tokenizer.fit_on_texts(list(data["question_text"]))

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



sequences = tokenizer.texts_to_sequences(data["question_text"])

maxlen = len(max(sequences, key = len)) #max number of words in a question (längste sequenz aus tokenisierten wörtern)



X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq =tokenizer.texts_to_sequences(X_test)

X_train_seq = pad_sequences(X_train_seq, maxlen=maxlen)

X_test_seq = pad_sequences(X_test_seq, maxlen=maxlen)
word = "kitchen"

print("The index of", word, "in the vocabulary is", word_index[word], ".")
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
oov = check_coverage(tokenizer.word_counts,embeddings_index)
oov[:10]
#dictionary

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
#Creation of the improved data

t1 = Tokenizer(filters = puncts,lower = False)

t1.fit_on_texts(list(data["question_text"]))

oov = check_coverage(t1.word_counts,embeddings_index)
oov[:10]
t2 = Tokenizer(filters = puncts, lower = True)

t2.fit_on_texts(list(data["question_text"]))

oov = check_coverage(t2.word_counts,embeddings_index)
oov[:10]
tokenizer_2 = Tokenizer(filters = puncts, lower = False)

tokenizer_2.fit_on_texts(list(data["question_text"]))

word_index_2 = tokenizer_2.word_index

print('Found %s unique tokens.' % len(word_index_2))



sequences_2 = tokenizer_2.texts_to_sequences(data["question_text"])

maxlen_2 = len(max(sequences_2, key = len)) #max number of words in a question (längste sequenz aus tokenisierten wörtern)



X_train_seq_2 = tokenizer_2.texts_to_sequences(X_train)

X_test_seq_2 =tokenizer_2.texts_to_sequences(X_test)

X_train_seq_2 = pad_sequences(X_train_seq_2, maxlen=maxlen_2)

X_test_seq_2 = pad_sequences(X_test_seq_2, maxlen=maxlen_2)
#Compute Embedding Matrix for the old data

embed_dim = 300 #da glove.840B.300d.txt bedeutet, dass 300d. vektor

embedding_matrix = np.zeros((len(word_index) + 1, embed_dim)) #creation of the numpy array

for word, i in tqdm(word_index.items()): #loop going through each word in word_index

    embedding_vector = embeddings_index.get(word) #for each word the programm takes the respective vector and calls it embedding_vector

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector # vector gets inserted into the numpy array at the place where the word would stand according to the index.
#Compute Embedding Matrix for the tweaked data

embed_dim = 300 #da glove.840B.300d.txt bedeutet, dass 300d. vektor

embedding_matrix_2 = np.zeros((len(word_index_2) + 1, embed_dim)) #creation of the numpy array

for word, i in tqdm(word_index_2.items()): #loop going through each word in word_index

    embedding_vector = embeddings_index.get(word) #for each word the programm takes the respective vector and calls it embedding_vector

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix_2[i] = embedding_vector # vector gets inserted into the numpy array at the place where the word would stand according to the index.
from keras.layers.embeddings import Embedding



#Load into Keras Embedding layer

embedding_layer = Embedding(len(word_index) + 1,

                            embed_dim,

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=False)
#Load into Keras Embedding layer

embedding_layer_2 = Embedding(len(word_index_2) + 1,

                            embed_dim,

                            weights=[embedding_matrix_2],

                            input_length=maxlen_2,

                            trainable=False)
#Metric: F1 score: F1: wikipedia, umsetzung https://github.com/keras-team/keras/blob/53e541f7bf55de036f4f5641bd2947b96dd8c4c3/keras/metrics.py



def fmeasure (y_true, y_pred):

    

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    

    f1 = 5 * (precision*recall) / (4*precision+recall+K.epsilon())

    

    return f1
#Metric: F1 score: F1: wikipedia, umsetzung https://github.com/keras-team/keras/blob/53e541f7bf55de036f4f5641bd2947b96dd8c4c3/keras/metrics.py



def loss_f1 (y_true, y_pred):

    

    true_positives = K.sum(y_true * y_pred)

    predicted_positives = K.sum(y_pred)

    possible_positives = K.sum(y_true)

    

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    

    f1 = 5 * (precision*recall) / (4*precision+recall+K.epsilon())

    

    return 1-f1
#Attention layer 

from keras import initializers, regularizers, constraints

from keras.engine.topology import Layer

import keras.backend as K #to use math functions like "keras.backend.sum"



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
from keras.models import Model #to build the Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation,Bidirectional, CuDNNGRU, CuDNNLSTM # the layers we will use
# First Model: Unprocessed Data, Basic Structure

sequence_input = Input(shape=(maxlen,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

X = LSTM(128, return_sequences=True)(embedded_sequences)

X = Dropout(0.5)(X)

X = LSTM(128)(X)

X = Dropout(0.5)(X)

X = Dense(1, activation = "sigmoid")(X)

X = Activation('sigmoid')(X)



model = Model(inputs=sequence_input, outputs=X)

model.summary()
# Second Model: Unprocessed Data, Improved Structure

sequence_input_2 = Input(shape=(maxlen,), dtype='int32')

embedded_sequences_2 = embedding_layer(sequence_input_2)

X_2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedded_sequences_2)

X_2 = Dropout(0.5)(X_2)

X_2 = Bidirectional(CuDNNGRU(128, return_sequences=False))(X_2)

X_2 = Dropout(0.5)(X_2)

X_2 = Dense(1)(X_2)

X_2 = Activation('sigmoid')(X_2)



model_2 = Model(inputs = sequence_input_2, outputs=X_2)

model_2.summary()
# Third Model: Preprocessed Data, Improved Structure

sequence_input_3 = Input(shape=(maxlen_2,), dtype='int32')

embedded_sequences_3 = embedding_layer_2(sequence_input_3)

X_3 = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedded_sequences_3)

X_3 = Dropout(0.5)(X_3)

X_3 = Bidirectional(CuDNNGRU(128, return_sequences=False))(X_3)

X_3 = Dropout(0.5)(X_3)

X_3 = Dense(1)(X_3)

X_3 = Activation('sigmoid')(X_3)



model_3 = Model(inputs = sequence_input_3, outputs=X_3)

model_3.summary()



model_5 = Model(inputs = sequence_input_3, outputs=X_3)
# Fourth Model: Preprocessed Data, With Attention Layer

sequence_input_4 = Input(shape=(maxlen_2,), dtype='int32')

embedded_sequences_4 = embedding_layer_2(sequence_input_4)

X_4 = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedded_sequences_4)

X_4 = Bidirectional(CuDNNGRU(64, return_sequences=True))(X_4)

X_4 = Attention(maxlen_2)(X_4)

X_4 = Dense(64, activation = "relu")(X_4)

X_4 = Dense(1, activation = "sigmoid")(X_4)



model_4 = Model(inputs = sequence_input_4, outputs = X_4)

model_4.summary()
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_fmeasure', min_delta=0.0001, patience=2, mode='max')

model.compile(loss= "binary_crossentropy", optimizer='adam', metrics= [fmeasure])

history = model.fit(X_train_seq, Y_train, validation_data=(X_test_seq, Y_test),

          epochs= 8, batch_size = 512, shuffle = True, callbacks = [early_stopping], verbose = 2)
model_2.compile(loss= "binary_crossentropy", optimizer='adam', metrics= [fmeasure])

history_2 = model_2.fit(X_train_seq, Y_train, validation_data=(X_test_seq, Y_test),

          epochs= 8, batch_size = 512, shuffle = True, callbacks = [early_stopping], verbose = 2)
model_3.compile(loss= "binary_crossentropy", optimizer='adam', metrics= [fmeasure])

history_3 = model_3.fit(X_train_seq_2, Y_train, validation_data=(X_test_seq_2, Y_test),

          epochs= 8, batch_size = 512, shuffle = True, callbacks = [early_stopping], verbose = 2)
model_4.compile(loss= "binary_crossentropy", optimizer='adam', metrics= [fmeasure])

history_4 = model_4.fit(X_train_seq_2, Y_train, validation_data=(X_test_seq_2, Y_test),

          epochs= 8, batch_size = 512, shuffle = True, callbacks = [early_stopping], verbose = 2)
# F1-Measure as loss-function

model_5.compile(loss= loss_f1, optimizer='adam', metrics= [fmeasure])

history_5 = model_5.fit(X_train_seq_2, Y_train, validation_data=(X_test_seq_2, Y_test),

          epochs= 8, batch_size = 512, shuffle = True, callbacks = [early_stopping], verbose = 2)
print(history.history.keys())

plt.plot(history.history["fmeasure"])

plt.plot(history.history["val_fmeasure"])

plt.title('Model 1')

plt.show()
print(history_2.history.keys())

plt.plot(history_2.history["fmeasure"])

plt.plot(history_2.history["val_fmeasure"])

plt.title('Model 2 -  with improved structure')

plt.show()
print(history_3.history.keys())

plt.plot(history_3.history["fmeasure"])

plt.plot(history_3.history["val_fmeasure"])

plt.title('Model 3 -  with improved structure and data preparation')

plt.show()
print(history_4.history.keys())

plt.plot(history_4.history["fmeasure"])

plt.plot(history_4.history["val_fmeasure"])

plt.title('Model 4 -  with data preparation and attention')

plt.show()
print(history_5.history.keys())

plt.plot(history_5.history["fmeasure"])

plt.plot(history_5.history["val_fmeasure"])

plt.title('Model 5 -  with improved structure, data preparation and f-measure as loss function')

plt.show()