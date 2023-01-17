# references:

# https://www.kaggle.com/antmarakis/cnn-baseline-model

# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

# https://github.com/yoonkim/CNN_sentence

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

seed = 12345

import random

import numpy as np

from tensorflow import set_random_seed



random.seed(seed)

np.random.seed(seed)

set_random_seed(seed)
train,test,sampleSubmission = pd.read_csv('../input/train.tsv', sep = '\t'),pd.read_csv('../input/test.tsv', sep = '\t'),pd.read_csv('../input/sampleSubmission.csv')



# train,test,sampleSubmission = pd.read_csv('all/train.tsv', sep = '\t'),pd.read_csv('all/test.tsv', sep = '\t'),pd.read_csv('all/sampleSubmission.csv')

train.head(3)

# From Yoon Kim work:

# https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

import re

def clean_str(string):

    """

    Tokenization/string cleaning for all datasets except for SST.

    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     

    string = re.sub(r"\'s", " \'s", string) 

    string = re.sub(r"\'ve", " \'ve", string) 

    string = re.sub(r"n\'t", " n\'t", string) 

    string = re.sub(r"\'re", " \'re", string) 

    string = re.sub(r"\'d", " \'d", string) 

    string = re.sub(r"\'ll", " \'ll", string) 

    string = re.sub(r",", " , ", string) 

    string = re.sub(r"!", " ! ", string) 

    string = re.sub(r"\(", " \( ", string) 

    string = re.sub(r"\)", " \) ", string) 

    string = re.sub(r"\?", " \? ", string) 

    string = re.sub(r"\s{2,}", " ", string)    

    return string.strip().lower()



phrases = [clean_str(s) for s in train['Phrase']]
type(train['Phrase'])

len(phrases[2].split())

phrases[2].split()
from keras.utils import to_categorical

Y = to_categorical(train['Sentiment'].values)

print(Y[155:165])

print(train['Sentiment'].values[155:165])
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# tokenizer = Tokenizer(num_words=max_features)

t = Tokenizer()

t.fit_on_texts(phrases)

vocab_size = len(t.word_index) + 1

X = t.texts_to_sequences(phrases)

# print(X)

max_length = max([len(test.split()) for test in phrases ])

X = pad_sequences(X,maxlen=max_length,padding = 'post')

# print(X)

print(X.shape)
from keras.layers import Embedding

from keras.models import Sequential, Model

from keras.layers import Dense, Activation

from keras.layers import Flatten, Conv1D, SpatialDropout1D, MaxPooling1D,AveragePooling1D, merge, concatenate, Input, Dropout



# ONE LAYER

def model(output_dim=8, max_length=50, y_dim=5, num_filters=5, filter_sizes = [3,5], pooling = 'max', pool_padding = 'valid', dropout = 0.2):

    # Input Layer

#     embed_input = Input(shape=(max_length,output_dim))

    embed_input = Input(shape=(max_length,))

    x = Embedding(vocab_size,output_dim,input_length=max_length)(embed_input)

#     x = SpatialDropout1D(0.2)(x)

    ## concat

    pooled_outputs = []

    for i in range(len(filter_sizes)):

        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)

        if pooling=='max':

            conv = MaxPooling1D(pool_size=max_length-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)

        else:

            conv = AveragePooling1D(pool_size=max_length-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)            

        pooled_outputs.append(conv)

    merge = concatenate(pooled_outputs)

        

    x = Flatten()(merge)

    x = Dropout(dropout)(x)

#     predictions = Dense(y_dim, activation = 'sigmoid')(x)

    predictions = Dense(y_dim, activation = 'softmax')(x) # TEST

    

    model = Model(inputs=embed_input,outputs=predictions)



    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

    print(model.summary())

    

    from keras.utils import plot_model

    plot_model(model, to_file='shared_input_layer.png')

    

    return model





model = model(output_dim=16, max_length=max_length,y_dim=5,filter_sizes = [3,4,5],pooling = 'max',dropout=0.5)

from IPython.display import Image

Image(filename='shared_input_layer.png') 

# ## PREVIOUS MODELS

# def model1(output_dim=8, max_length=50, y_dim=5, filter_sizes = [3]):

#     model = Sequential()

#     model.add(Embedding(vocab_size,output_dim,input_length=max_length))

#     model.add(Flatten())

#     model.add(Dense(y_dim, activation = 'sigmoid'))



#     model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

#     print(model.summary())

#     return model



# def model2(output_dim=8, max_length=50, y_dim=5, num_filters=5, filter_sizes = [3,5]):

#     model = Sequential()

#     model.add(Embedding(vocab_size,output_dim,input_length=max_length))



#     model.add(SpatialDropout1D(0.2))



#     ## GOOD

#     model.add(Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', activation='relu'))

#     model.add(MaxPooling1D(pool_size=max_length-filter_sizes[0]+1, strides=1, padding='valid'))



#     model.add(Flatten())

#     model.add(Dense(y_dim, activation = 'sigmoid'))



#     model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

#     print(model.summary())

#     return model



from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)
epochs = 10

batch_size = 32



model.fit(X_train,Y_train,epochs = epochs, validation_data=(X_val,Y_val), batch_size=batch_size, verbose = 1)

loss,accuracy = model.evaluate(X_val,Y_val)

print('Accuracy: %f' % (accuracy*100))
# epochs = 10

# batch_size = 32



# model_avg = model(output_dim=16, max_length=max_length,y_dim=5,filter_sizes = [3,4,5],pooling = 'avg',dropout=0.5)



# model_avg.fit(X_train,Y_train,epochs = epochs, validation_data=(X_val,Y_val), batch_size=batch_size, verbose = 1)

# loss,accuracy = model_avg.evaluate(X_val,Y_val)

# print('Accuracy: %f' % (accuracy*100))

# ONE LAYER

def model_constant_poolsize(output_dim=8, max_length=50, y_dim=5, num_filters=5, filter_sizes = [3,5], pooling = 'max', pool_padding = 'valid', pool_size=5,dropout=0.4):

    # Input Layer

#     embed_input = Input(shape=(max_length,output_dim))

    embed_input = Input(shape=(max_length,))

    x = Embedding(vocab_size,output_dim,input_length=max_length)(embed_input)

#     x = SpatialDropout1D(0.2)(x)

    ## concat

    pooled_outputs = []

    for i in range(len(filter_sizes)):

        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='same', activation='relu')(x)

        if pooling=='max':

            conv = MaxPooling1D(pool_size=pool_size, strides=1, padding = pool_padding)(conv)

        else:

            conv = AveragePooling1D(pool_size=pool_size, strides=1, padding = pool_padding)(conv)            

        pooled_outputs.append(conv)

    merge = concatenate(pooled_outputs)

        

    x = Flatten()(merge)

    x = Dropout(dropout)(x)

#     predictions = Dense(y_dim, activation = 'sigmoid')(x)

    predictions = Dense(y_dim, activation = 'softmax')(x) # TEST

    

    model = Model(inputs=embed_input,outputs=predictions)



    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

    print(model.summary())

    

    from keras.utils import plot_model

    plot_model(model, to_file='shared_input_layer_model_constant_poolsize.png')

    

    return model





model_constant_poolsize = model_constant_poolsize(output_dim=16, max_length=max_length,y_dim=5,filter_sizes = [3,4,5],pooling = 'max', pool_size=5,dropout = 0.5)

from IPython.display import Image

Image(filename='shared_input_layer_model_constant_poolsize.png') 

epochs = 10

model_constant_poolsize.fit(X_train,Y_train,epochs = epochs, validation_data=(X_val,Y_val), batch_size=batch_size, verbose = 1)

loss,accuracy = model_constant_poolsize.evaluate(X_val,Y_val)

print('Accuracy: %f' % (accuracy*100))

# pre processing

test_phrases = [clean_str(s) for s in test['Phrase']]

# text to tokens

X_test = t.texts_to_sequences(test_phrases)

X_test = pad_sequences(X_test,maxlen=max_length,padding = 'post')
# sampleSubmission['Sentiment'] = model.predict_classes(X_test,verbose=1)

sampleSubmission['Sentiment'] = model_constant_poolsize.predict(X_test,verbose=1).argmax(axis=-1)

sampleSubmission.to_csv('sub_cnn_constant_maxpool.csv', index=False)





sampleSubmission['Sentiment'] = model.predict(X_test,verbose=1).argmax(axis=-1)

sampleSubmission.to_csv('sub_cnn.csv', index=False)
# train

model_constant_poolsize.fit(X,Y,epochs = epochs, validation_data=(X_val,Y_val), batch_size=batch_size, verbose = 1)



# submission

sampleSubmission['Sentiment'] = model_constant_poolsize.predict(X_test,verbose=1).argmax(axis=-1)

sampleSubmission.to_csv('sub_cnn_constant_maxpool_FULL.csv', index=False)