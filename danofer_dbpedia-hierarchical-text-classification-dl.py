import spacy



import pandas as pd

import os

import numpy as np



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence



# fix random seed for reproducibility

np.random.seed(7)



## For basic data preproc: https://realpython.com/python-keras-text-classification/

from keras.preprocessing.text import Tokenizer



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.sequence import pad_sequences



from keras.utils.np_utils import to_categorical # https://stackoverflow.com/questions/36927025/how-to-use-keras-multi-layer-perceptron-for-multi-class-classification



from keras.models import Sequential

from keras import layers



from keras.layers import Embedding,GlobalMaxPool1D,Dense, Conv1D, LSTM, Dropout, SpatialDropout1D, Input, merge, Multiply, Dot

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Bidirectional, MaxPooling1D

from keras import regularizers, optimizers



from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.initializers import Constant

from keras.models import Model

from keras import optimizers



pd.set_option('max_colwidth',400)



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

import re



from sklearn.utils import shuffle

from tqdm import tqdm

#nltk.download('stopwords')

from sklearn import metrics



## https://www.kaggle.com/nikhilroxtomar/lstm-cnn-1d-with-lstm-attention



from keras.models import Model

from keras.layers import Input, Dense, Embedding, concatenate

from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D

from keras.layers import Add, BatchNormalization, Activation, CuDNNLSTM, Dropout

from keras.layers import *

from keras.models import *

from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
print(os.listdir("../input"))

print(os.listdir("../input/concept-net-numberbatch"))

print(os.listdir("../input/dbpedia-classes"))
TARGET_COL = "l1"



maxlen = 75

EMBEDDING_DIR = "../input/concept-net-numberbatch"

EMBEDDING_FILE = "numberbatch-en.txt"

EMBEDDING_DIM = 100



# TRAIN_FILE = os.path.join("../input","dbpedia_train.csv")

# VAL_FILE = os.path.join("../input","dbpedia_val.csv")



TRAIN_FILE = "../input/dbpedia-classes/DBPEDIA_train.csv"

VAL_FILE = "../input/dbpedia-classes/DBPEDIA_val.csv"
## https://realpython.com/python-keras-text-classification/

import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
### https://www.kaggle.com/fareise/multi-head-self-attention-for-text-classification



class Position_Embedding(Layer):

    

    def __init__(self, size=None, mode='sum', **kwargs):

        self.size = size

        self.mode = mode

        super(Position_Embedding, self).__init__(**kwargs)

        

    def call(self, x):

        if (self.size == None) or (self.mode == 'sum'):

            self.size = int(x.shape[-1])

        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]

        position_j = 1. / K.pow(10000., \

                                 2 * K.arange(self.size / 2, dtype='float32' \

                               ) / self.size)

        position_j = K.expand_dims(position_j, 0)

        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 

        position_i = K.expand_dims(position_i, 2)

        position_ij = K.dot(position_i, position_j)

        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)

        if self.mode == 'sum':

            return position_ij + x

        elif self.mode == 'concat':

            return K.concatenate([position_ij, x], 2)

        

    def compute_output_shape(self, input_shape):

        if self.mode == 'sum':

            return input_shape

        elif self.mode == 'concat':

            return (input_shape[0], input_shape[1], input_shape[2]+self.size)





'''

output dimention: [batch_size, time_step, nb_head*size_per_head]

every word can be represented as a vector [nb_head*size_per_head]

'''

class Attention(Layer):



    def __init__(self, nb_head, size_per_head, **kwargs):

        self.nb_head = nb_head

        self.size_per_head = size_per_head

        self.output_dim = nb_head*size_per_head

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        self.WQ = self.add_weight(name='WQ', 

                                  shape=(input_shape[0][-1], self.output_dim),

                                  initializer='glorot_uniform',

                                  trainable=True)

        self.WK = self.add_weight(name='WK', 

                                  shape=(input_shape[1][-1], self.output_dim),

                                  initializer='glorot_uniform',

                                  trainable=True)

        self.WV = self.add_weight(name='WV', 

                                  shape=(input_shape[2][-1], self.output_dim),

                                  initializer='glorot_uniform',

                                  trainable=True)

        super(Attention, self).build(input_shape)

        

    def Mask(self, inputs, seq_len, mode='mul'):

        if seq_len == None:

            return inputs

        else:

            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])

            mask = 1 - K.cumsum(mask, 1)

            for _ in range(len(inputs.shape)-2):

                mask = K.expand_dims(mask, 2)

            if mode == 'mul':

                return inputs * mask

            if mode == 'add':

                return inputs - (1 - mask) * 1e12

                

    def call(self, x):

        if len(x) == 3:

            Q_seq,K_seq,V_seq = x

            Q_len,V_len = None,None

        elif len(x) == 5:

            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)

        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))

        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))

        K_seq = K.dot(K_seq, self.WK)

        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))

        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))

        V_seq = K.dot(V_seq, self.WV)

        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))

        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5

        A = K.permute_dimensions(A, (0,3,2,1))

        A = self.Mask(A, V_len, 'add')

        A = K.permute_dimensions(A, (0,3,2,1))    

        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])

        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))

        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))

        O_seq = self.Mask(O_seq, Q_len, 'mul')

        return O_seq

        

    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], input_shape[0][1], self.output_dim)
train = pd.read_csv(TRAIN_FILE,usecols=["text",TARGET_COL]).sample(frac=0.12) # downsample for speed

val =  pd.read_csv(VAL_FILE,usecols=["text",TARGET_COL]).sample(frac=0.5)

print(train.shape)

print(val.shape)

train.tail()
N_CLASSES = train[TARGET_COL].nunique()
encoder = LabelEncoder()

y_train = encoder.fit_transform(train[TARGET_COL])

y_val = encoder.transform(val[TARGET_COL]) # transform, not fit! 







# # onehot encode ### fix these if changing from train_labels to y_train .. 

# onehot_encoder = OneHotEncoder(sparse=False)

# integer_encoded = integer_encoded.reshape(len(train_labels), 1)

# onehot_encoded = onehot_encoder.fit_transform(train_labels)
## https://stackoverflow.com/questions/36927025/how-to-use-keras-multi-layer-perceptron-for-multi-class-classification



# y_train = to_categorical(train[TARGET_COL])

# y_val = to_categorical(val[TARGET_COL])



y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
# lb = LabelBinarizer()

# y_train = lb.fit_transform(train[TARGET_COL])



# y_val = lb.transform(val[TARGET_COL])
sentences_train = train["text"].values

sentences_val = val["text"].values
## https://realpython.com/python-keras-text-classification/



tokenizer = Tokenizer(num_words=151234)

tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)

X_val = tokenizer.texts_to_sequences(sentences_val)



vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index



print(sentences_train[2])

print(X_train[2])
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

embeddings_index = {}



f = open(os.path.join(EMBEDDING_DIR, EMBEDDING_FILE))

# f = open("../input/concept-net-numberbatch/numberbatch-en.txt")



for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:(EMBEDDING_DIM+1)], dtype='float32') # take first 100 dims only

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
## At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix:



embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
## define pretrained embedding layer, not trainable



embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=False)
## set up callbacks and optimizer defaults

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=3, min_lr=0.004)



earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,restore_best_weights=True)

# model.fit(X_train, Y_train, callbacks=[reduce_lr])





adam = optimizers.Adam(lr=0.07,decay=0.0005)
#### https://www.kaggle.com/fareise/multi-head-self-attention-for-text-classification



config = {

    "trainable": False,

    "max_len": maxlen,

    "max_features": 95000,

    "embed_size": EMBEDDING_DIM,

    "units": 128,

    "num_heads": 8,

    "dr": 0.1,

    "epochs": 2,

    "model_checkpoint_path": "best_weights",

}

def build_model(config):

    inp = Input(shape = (config["max_len"],))

    

    x = embedding_layer(inp)

    x = Position_Embedding()(x)

    x = Attention(config["num_heads"], config["units"])([x, x, x])  #output: [batch_size, time_step, nb_head*size_per_head]

    x = GlobalMaxPooling1D()(x)

#     x = Dropout(config["dr"])(x)

    x = Dense(N_CLASSES, activation='softmax')(x)



    

    model = Model(inputs = inp, outputs = x)

    model.compile(

        loss = "categorical_crossentropy", 

        #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 

        optimizer = "adam",

        metrics = ["accuracy"])

    

    return model



model = build_model(config)



# ### model train + history + run



history = model.fit(X_train, y_train,

                    epochs=2,

                    verbose=1,

#                     validation_data=(X_test, y_test),

                    validation_split=0.1,

                    batch_size=128)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val, y_val, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)





score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])
## attention based biLSTM:

### https://www.kaggle.com/danofer/different-embeddings-with-attention-fork

## https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-672-lb

## use GRU for now





# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043

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

    

    



inp = Input(shape=(maxlen,))

x = embedding_layer(inp)

x = Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)

x = Bidirectional(GRU(50, return_sequences=True))(x)

x = Attention(maxlen)(x)

x = Dense(64, activation="relu")(x)

x = Dense(N_CLASSES, activation='softmax')(x)



model = Model(inputs = inp, outputs = x)

model.compile(

    loss = "categorical_crossentropy", 

    #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 

    optimizer = "adam",

    metrics = ["accuracy"])





# model = build_model(config)



# ### model train + history + run



history = model.fit(X_train, y_train,

                    epochs=2,

                    verbose=1,

#                     validation_data=(X_test, y_test),

                    validation_split=0.1,

                    batch_size=64)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val, y_val, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)





score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])

### https://www.kaggle.com/nikhilroxtomar/lstm-cnn-1d-with-lstm-attention





## https://www.kaggle.com/nikhilroxtomar/lstm-cnn-1d-with-lstm-attention



def attention_3d_block(inputs, name):

    # inputs.shape = (batch_size, time_steps, input_dim)

    print("inputs.shape[1].value",inputs.shape[1].value)

    TIME_STEPS = inputs.shape[1].value

    SINGLE_ATTENTION_VECTOR = False

    

    input_dim = int(inputs.shape[2])

    print("input_dim.shape",input_dim.shape)

    a = Permute((2, 1))(inputs)

    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(TIME_STEPS, activation='softmax')(a)

    if SINGLE_ATTENTION_VECTOR:

        a = Lambda(lambda x: K.mean(x, axis=1))(a)

        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name=name)(a)

    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul





def model1(init):

    x = init

    x = Conv1D(64, 3,strides=2,padding='same',activation='relu')(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x) #CuDNN

    x = attention_3d_block(x, 'attention_vec_1')

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x) #CuDNN

    x = attention_3d_block(x, 'attention_vec_2')

    x = GlobalMaxPool1D()(x)

    out = Dense(64, activation="relu")(x)

    return out



def m2_block(init, filter, kernel, pool):

    x = init

    

    x = Conv1D(filter, kernel, padding='same', kernel_initializer='he_normal', activation='elu')(x)

    skip = x

    x = Conv1D(filter, kernel, padding='same', kernel_initializer='he_normal', activation='elu')(x)

    x = Conv1D(filter, kernel, padding='same')(x)

    x = BatchNormalization()(x)

    x = Add()([x, skip])

    x = Activation('elu')(x) # vs relu

    x = MaxPooling1D(pool)(x)

    

    x = Flatten()(x)

    x = BatchNormalization()(x)

    

    return x



def model2(init):

    #init = Reshape((maxlen, embed_size, 1))(init)

    

    # pool = maxlen - filter + 1

    x0 = m2_block(init, 32, 1, maxlen - 1 + 1)

    x1 = m2_block(init, 32, 2, maxlen - 2 + 1)

    x2 = m2_block(init, 32, 3, maxlen - 3 + 1)

    x3 = m2_block(init, 32, 5, maxlen - 5 + 1)

    

    x = concatenate([x0, x1, x2, x3])

    x = Dropout(0.1)(x)  # vs 0.5

    out = Dense(64, activation="relu")(x)

    return out





def get_model():

    inp = Input(shape=(maxlen, ))

    #x = Embedding(max_features, embed_size)(inp)

#     x = Embedding(input_dim=max_features, output_dim= embed_size , input_length=maxlen,weights=[embedding_matrix], trainable=False)(inp)

    x = embedding_layer(inp)

    

    out1 = model1(x)

    out2 = model2(x)

    

    conc = concatenate([out1, out2])

    

    #conc = out1

    x = Dropout(0.15)(conc)  # vs 0.5

    x = Dense(64, activation='elu')(x) # relu

    x = Reshape((x.shape[1].value, 1))(x)

    x = CuDNNLSTM(32)(x)



    x = Dense(64, activation="elu")(x) # relu

    outp = Dense(N_CLASSES, activation='softmax')(x)

    

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='categorical_crossentropy',

                  optimizer='adam',

                  metrics=['acc'])    



    return model



# model = get_model()



#################

inp = Input(shape=(maxlen, ))

#x = Embedding(max_features, embed_size)(inp)

#     x = Embedding(input_dim=max_features, output_dim= embed_size , input_length=maxlen,weights=[embedding_matrix], trainable=False)(inp)

x = embedding_layer(inp)



out1 = model1(x)

out2 = model2(x)



conc = concatenate([out1, out2])



#conc = out1

x = Dropout(0.15)(conc)  # vs 0.5

x = Dense(64, activation='elu')(x) # relu

x = Reshape((x.shape[1].value, 1))(x)

#     x = CuDNNLSTM(32)(x)

x = LSTM(32)(x)



x = Dense(64, activation="elu")(x) # relu

outp = Dense(N_CLASSES, activation='softmax')(x)



model = Model(inputs=inp, outputs=outp)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])    

#################





model.summary()







history = model.fit(X_train, y_train,

                    epochs=2,

                    verbose=1,

#                     validation_split=0.1,

                    batch_size=128)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val, y_val, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)





score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])

# ### huge speed difference if embedding is not trainable..



# # embedding_dim = 80



# model = Sequential()

# # model.add(layers.Dropout(0.2))

# model.add(embedding_layer)

# model.add(layers.GlobalMaxPool1D())

# model.add(layers.Flatten()) # gives error

# # model.add(layers.Dropout(0.35))

# model.add(layers.Dense(128, activation='relu'))

# # model.add(layers.Dense(N_CLASSES, activation='relu'))

# model.add(layers.Dense(N_CLASSES, activation='softmax')) # Instead of 1 sigmoid output (binary classification)

# model.compile(optimizer='adam',

#               loss='categorical_crossentropy',

#               metrics=['accuracy'])

# model.summary()
# ### model train + history + run



# history = model.fit(X_train, y_train,

#                     epochs=5,

#                     verbose=1,

# #                     validation_data=(X_test, y_test),

#                     validation_split=0.1,

#                     batch_size=128)

# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

# print("Training Accuracy: {:.4f}".format(accuracy))

# loss, accuracy = model.evaluate(X_val, y_val, verbose=False)

# print("Testing Accuracy:  {:.4f}".format(accuracy))

# plot_history(history)





# score = model.evaluate(X_val, y_val,

#                        verbose=1)

# print('Test score:', score[0])

# print('Test accuracy:', score[1])
## basic 1D CNN:



# embedding_dim = 80



model = Sequential()

# model.add(layers.Dropout(0.1))

# model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen,

#                           trainable=False))

model.add(embedding_layer)



# model.add(layers.Dropout(0.1))

model.add(layers.Conv1D(128, 5, activation='relu'))

# model.add(layers.GlobalMaxPooling1D())# this being activated gives bug..

# model.add(layers.Conv1D(32, 5, activation='relu')) # added

model.add(layers.GlobalMaxPooling1D()) # added

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.1))

model.add(layers.Dense(32, activation='relu'))

# model.add(layers.Dropout(0.1))

# model.add(layers.Dense(1, activation='sigmoid')) #orig

model.add(layers.Dense(N_CLASSES, activation='softmax')) #new



model.compile(optimizer=adam, #'adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])







### model train + history + run



history = model.fit(X_train, y_train,

                    epochs=2,

                    verbose=1,

#                     validation_data=(X_test, y_test),

                    validation_split=0.1,

                    batch_size=128

                   , callbacks=[reduce_lr,earlystop])



model.summary()



score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])
## Another  1D CNN:

##TODO: use pretrained embeddings : https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py





# embedding_dim = 50



model = Sequential()

# model.add(layers.Dropout(0.1))

# model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen

#                           ,trainable=False

#                           ))



model.add(embedding_layer)



# model.add(layers.Dropout(0.1))

model.add(layers.Conv1D(128, 5, activation='relu'))



model.add(MaxPooling1D(2))

# model.add(layers.Dropout(0.1))



model.add(layers.Conv1D(64, 5, activation='relu'))



# model.add(MaxPooling1D(2))

# model.add(layers.Dropout(0.1))



# model.add(layers.Conv1D(64, 5, activation='relu'))



# model.add(layers.GlobalMaxPooling1D())# this being activated gives bug..

# model.add(layers.Conv1D(32, 5, activation='relu')) # added

model.add(layers.GlobalMaxPooling1D()) # added

# model.add(layers.Dropout(0.2))

model.add(layers.Dense(32, activation='relu'))

# model.add(layers.Dropout(0.1))

# model.add(layers.Dense(1, activation='sigmoid')) #orig

model.add(layers.Dense(N_CLASSES, activation='softmax')) #new



model.compile(optimizer=adam, #'adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])







### model train + history + run



history = model.fit(X_train, y_train,

                    epochs=2,

                    verbose=1,

#                     validation_data=(X_test, y_test),

                    validation_split=0.1,

                    batch_size=128

                   , callbacks=[reduce_lr,earlystop])



model.summary()



score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])
## Another  1D CNN:

##TODO: use pretrained embeddings : https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py







model = Sequential()

# model.add(layers.Dropout(0.1))

model.add(embedding_layer)

# model.add(layers.Dropout(0.1))

model.add(layers.Conv1D(128, 4, activation='relu',padding='valid'))



# model.add(MaxPooling1D(2))

# model.add(layers.Dropout(0.1))



model.add(layers.Conv1D(64, 6, activation='relu'))



# model.add(MaxPooling1D(2))

# # model.add(layers.Dropout(0.1))



# model.add(layers.Conv1D(64, 4, activation='relu'))

# # model.add(MaxPooling1D(2))



# model.add(layers.Conv1D(32, 5, activation='relu')) # added



# model.add(layers.GlobalMaxPooling1D()) # added



model.add(layers.Flatten())

# model.add(layers.Dropout(0.1))

model.add(layers.Dense(128, activation='relu'))

# model.add(layers.Dense(128, activation='relu'))

# model.add(layers.Dropout(0.2))

model.add(layers.Dense(N_CLASSES, activation='softmax')) #new



model.compile(optimizer=adam, #'adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.summary()



### model train + history + run



history = model.fit(X_train, y_train,

                    epochs=3,

                    verbose=1,

#                     validation_data=(X_test, y_test),

                    validation_split=0.1,

                    batch_size=128

                   , callbacks=[reduce_lr,earlystop])







score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])
## imdb cnn-lstm example

## https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py

## bi-lstm

##https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py



model = Sequential()

model.add(embedding_layer)

model.add(Bidirectional(LSTM(50)))

# model.add(Dropout(0.2))

model.add(layers.Dense(N_CLASSES, activation='softmax')) #new





# try using different optimizers and different optimizer configs



model.compile(optimizer=adam, #'adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])





model.summary()





### model train + history + run



history = model.fit(X_train, y_train,

                    epochs=2,

                    verbose=1,

                    validation_split=0.1,

                    batch_size=64

                   , callbacks=[reduce_lr,earlystop])



score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])

## defaultish network https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html



sequence_input = Input(shape=(maxlen,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = GlobalMaxPooling1D()(x)  # global max pooling

x = Flatten()(x)

x = Dense(128, activation='relu')(x)

preds = Dense(N_CLASSES, activation='softmax')(x)



model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



model.fit(X_train, y_train, verbose=1,

                    validation_split=0.1,

          epochs=2,

          batch_size=128)





score = model.evaluate(X_val, y_val,

                       verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])

# # #### attention 

# # ## https://github.com/philipperemy/keras-attention-mechanism



# ### older keras version, update to use merge/multiply layer



# # inputs = Input(shape=(input_dim,))



# # # ATTENTION PART STARTS HERE

# # attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)

# # attention_mul = merge([inputs, attention_probs], output_shape=32, name='attention_mul', mode='mul')

# # # ATTENTION PART FINISHES HERE



# # attention_mul = Dense(64)(attention_mul)

# # output = Dense(1, activation='sigmoid')(attention_mul)

# # model = Model(input=[inputs], output=output)







# ### https://www.reddit.com/r/learnmachinelearning/comments/adhsfm/keras_mergelayer1_layer2_modemul/

# # merged = keras.layers.Multiply()([tanh_out, sigmoid_out])

# # Here merged is actually a layer so first you're creating a Multiply object and then calling it. It would be equivalent to this:

# import keras

# multiply_layer = keras.layers.Multiply()

# # multiply_layer = keras.layers.dot()

# # merged = multiply_layer([layer1, layer2])





# sequence_input = Input(shape=(maxlen,), dtype='int32')

# embedded_sequences = embedding_layer(sequence_input)

# seq_v = keras.layers.Flatten()(embedded_sequences)

# attention_probs = Dense(10500, activation='softmax', name='attention_vec')(seq_v)

# # attention_mul = merge([embedded_sequences, attention_probs], output_shape=maxlen, name='attention_mul', mode='mul')

# # https://www.reddit.com/r/learnmachinelearning/comments/adhsfm/keras_mergelayer1_layer2_modemul/

# # attention_mul = multiply_layer([embedded_sequences, attention_probs])

# # attention_mul = keras.layers.dot([seq_v, attention_probs],axes=0,normalize=False)



# attention_mul = multiply_layer([seq_v, attention_probs])





# # attention_mul = Flatten(attention_mul)

# attention_mul = Dense(64)(attention_mul)



# preds = Dense(N_CLASSES, activation='softmax')(attention_mul)



# model = Model(sequence_input, preds)

# model.compile(loss='categorical_crossentropy',

#               optimizer='adam',

#               metrics=['accuracy'])



# model.summary()



# model.fit(X_train, y_train, verbose=1,

#                     validation_split=0.1,

#           epochs=3, batch_size=128)

# #### attention 

# ## https://github.com/philipperemy/keras-attention-mechanism



### older keras version, update to use merge/multiply layer



# inputs = Input(shape=(input_dim,))



# # ATTENTION PART STARTS HERE

# attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)

# attention_mul = merge([inputs, attention_probs], output_shape=32, name='attention_mul', mode='mul')

# # ATTENTION PART FINISHES HERE



# attention_mul = Dense(64)(attention_mul)

# output = Dense(1, activation='sigmoid')(attention_mul)

# model = Model(input=[inputs], output=output)







### https://www.reddit.com/r/learnmachinelearning/comments/adhsfm/keras_mergelayer1_layer2_modemul/

# merged = keras.layers.Multiply()([tanh_out, sigmoid_out])

# Here merged is actually a layer so first you're creating a Multiply object and then calling it. It would be equivalent to this:

import keras

multiply_layer = keras.layers.Multiply()

# multiply_layer = keras.layers.dot()

# merged = multiply_layer([layer1, layer2])





sequence_input = Input(shape=(maxlen,), dtype='int32')

seq_v = embedding_layer(sequence_input)

# seq_v = keras.layers.Flatten()(seq_v)

attention_probs = Dense(maxlen, activation='softmax', name='attention_vec')(seq_v)

# attention_mul = merge([embedded_sequences, attention_probs], output_shape=maxlen, name='attention_mul', mode='mul')

# https://www.reddit.com/r/learnmachinelearning/comments/adhsfm/keras_mergelayer1_layer2_modemul/

# attention_mul = multiply_layer([embedded_sequences, attention_probs])

# attention_mul = keras.layers.dot([seq_v, attention_probs],axes=0,normalize=False)



attention_mul = multiply_layer([seq_v, attention_probs])





# attention_mul = Flatten(attention_mul)

attention_mul = Dense(64)(attention_mul)



preds = Dense(N_CLASSES, activation='softmax')(attention_mul)



model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()



model.fit(X_train, y_train, verbose=1,

                    validation_split=0.1,

          epochs=1,

          batch_size=128)
