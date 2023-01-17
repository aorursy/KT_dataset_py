#model to classify bias unbiased
from keras.layers import Input, Embedding, LSTM, Dense,concatenate,GlobalMaxPool1D, Dropout, BatchNormalization,Flatten,Multiply
from keras.models import Model
#from attention_decoder import AttentionDecoder
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import glob
import re, os, codecs
import time
from keras.utils import to_categorical
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback
#from keras import initializations
from keras import initializers, regularizers, constraints
from time import sleep
from sklearn.model_selection import train_test_split
from tqdm import tqdm
class TimerCallback(Callback):
    
    def __init__(self, maxExecutionTime, byBatch = False, on_interrupt=None):
        
# Arguments:
#     maxExecutionTime (number): Time in minutes. The model will keep training 
#                                until shortly before this limit
#                                (If you need safety, provide a time with a certain tolerance)

#     byBatch (boolean)     : If True, will try to interrupt training at the end of each batch
#                             If False, will try to interrupt the model at the end of each epoch    
#                            (use `byBatch = True` only if each epoch is going to take hours)          

#     on_interrupt (method)          : called when training is interrupted
#         signature: func(model,elapsedTime), where...
#               model: the model being trained
#               elapsedTime: the time passed since the beginning until interruption   

        
        self.maxExecutionTime = maxExecutionTime * 60
        self.on_interrupt = on_interrupt
        
        #the same handler is used for checking each batch or each epoch
        if byBatch == True:
            #on_batch_end is called by keras every time a batch finishes
            self.on_batch_end = self.on_end_handler
        else:
            #on_epoch_end is called by keras every time an epoch finishes
            self.on_epoch_end = self.on_end_handler
    
    
    #Keras will call this when training begins
    def on_train_begin(self, logs):
        self.startTime = time.time()
        self.longestTime = 0            #time taken by the longest epoch or batch
        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished
    
    
    #this is our custom handler that will be used in place of the keras methods:
        #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`
    def on_end_handler(self, index, logs):
        
        currentTime      = time.time()                           
        self.elapsedTime = currentTime - self.startTime    #total time taken until now
        thisTime         = currentTime - self.lastTime     #time taken for the current epoch
                                                               #or batch to finish
        
        self.lastTime = currentTime
        
        #verifications will be made based on the longest epoch or batch
        if thisTime > self.longestTime:
            self.longestTime = thisTime
        
        
        #if the (assumed) time taken by the next epoch or batch is greater than the
            #remaining time, stop training
        remainingTime = self.maxExecutionTime - self.elapsedTime
        if remainingTime < self.longestTime:
            
            self.model.stop_training = True  #this tells Keras to not continue training
            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")
            
            #if we have passed the `on_interrupt` callback, call it here
            if self.on_interrupt is not None:
                self.on_interrupt(self.model, self.elapsedTime)
def saveWeights(model, elapsed):
    print ("Saving Weights")
    model.save_weights("model_weights.h5")
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
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
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
df = pd.read_csv("../input/Sentiment_Analysis_Dataset_Final.csv")
df.loc[(df['sentiment']==4),'sentiment']=1
df.head()
embed_size=300 #ideally around 30
time_steps=20
num_classes=2
unit_length=32
MAX_NB_WORDS = 200000
print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(df['text'])  #leaky
sequences = tokenizer.texts_to_sequences(df['text'])
word_index = tokenizer.word_index
padded_docs = pad_sequences(sequences, maxlen=time_steps, padding='post')
print("dictionary size: ", len(word_index))
labels = to_categorical(df['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.1, random_state=42)
inputs = Input(shape=(time_steps,))
#x=Embedding(input_dim=nb_words, output_dim=embed_size,weights=[embedding_matrix], input_length=time_steps, trainable=False)(inputs)
x = Embedding(output_dim=embed_size, input_dim=len(word_index), input_length=time_steps)(inputs)
y1= LSTM(unit_length,return_sequences=True,go_backwards=True)(x)
#y1 = GlobalMaxPool1D()(y1)
y1= Dropout(0.5)(y1)
y1 = BatchNormalization()(y1)
y2= LSTM(unit_length,return_sequences=True)(x)
y2= Dropout(0.5)(y2)
z1= LSTM(unit_length,return_sequences=True,go_backwards=True)(y2)
z1= Dropout(0.5)(z1)
z2= LSTM(unit_length,return_sequences=True)(y1)
z2= Dropout(0.2)(z2)
a1= LSTM(unit_length,return_sequences=True,go_backwards=True)(z2)
a2= LSTM(unit_length,return_sequences=True)(z1)
b= concatenate([a1,a2],axis=1)
att = Attention(2*time_steps)(b)
#d1 = Flatten()(att)
#d1 = Dense(512, activation="relu")(a1)
#d1 = Dense(512, activation="relu")(d1)
#d1 = Dense(256, activation="relu")(d1)
#d1 = Dense(128, activation="relu")(d1)
d1 = Dense(64, activation="relu")(att)
d1 = Dense(64, activation="relu")(d1)
d1 = Dropout(0.2)(d1)
main_output = Dense(num_classes, activation='softmax')(d1)
callbacks = [TimerCallback(340, byBatch=True,on_interrupt=saveWeights)]
model = Model(inputs=inputs, outputs=main_output)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train,epochs=2,callbacks=callbacks)
preds = model.evaluate(X_test[:5000], y_test[:5000])
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
"""
model1 = Model(inputs=inputs, outputs=main_output)
model1.load_weights("model_weights.h5")
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
preds = model1.evaluate(X_test[:5000], y_test[:5000])
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
"""

