import os

import re

from glob import glob

from tqdm import tqdm

import numpy as np

import pandas as pd

import ast

import matplotlib.pyplot as plt

%matplotlib inline
fnames = glob('../input/train_simplified/*.csv') #<class 'list'>

cnames = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']

drawlist = []

for f in fnames[0:6]: # num of word : 5

    first = pd.read_csv(f, nrows=10) # make sure we get a recognized drawing

    first = first[first.recognized==True].head(2) # top head 2 get 

    drawlist.append(first)

draw_df = pd.DataFrame(np.concatenate(drawlist), columns=cnames) # <class 'pandas.core.frame.DataFrame'>

draw_df
draw_df.drawing.values[0]
evens = range(0,11,2)

odds = range(1,12, 2)

# We have drawing images, 2 per label, consecutively

df1 = draw_df[draw_df.index.isin(evens)]

df2 = draw_df[draw_df.index.isin(odds)]



example1s = [ast.literal_eval(pts) for pts in df1.drawing.values]

example2s = [ast.literal_eval(pts) for pts in df2.drawing.values]

labels = df2.word.tolist()



for i, example in enumerate(example1s):

    plt.figure(figsize=(6,3))

    

    for x,y in example:

        plt.subplot(1,2,1)

        plt.plot(x, y, marker='.')

        plt.axis('off')



    for x,y, in example2s[i]:

        plt.subplot(1,2,2)

        plt.plot(x, y, marker='.')

        plt.axis('off')

        label = labels[i]

        plt.title(label, fontsize=10)



    plt.show()  
%reset -f


import os

from glob import glob

import re

import ast

import numpy as np 

import pandas as pd

from PIL import Image, ImageDraw 

from tqdm import tqdm

from dask import bag

import json



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from keras.models import Model

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.metrics import top_k_categorical_accuracy

from keras.layers import Input, Conv1D, Dense, Dropout, BatchNormalization, Flatten, MaxPool1D

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
path = '../input/train_simplified/'

classfiles = os.listdir(path)



numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} # sleeping bag -> sleeping_bag

files = [os.path.join(path, file) for i, file in enumerate(classfiles)]

word_mapping = {file.split('/')[-1][:-4]:i for i, file in enumerate(files)}



num_classes = len(files)    #340

imheight, imwidth = 32, 32 # size of an image

ims_per_class = 2000  #max? # in the code above and above, there existed more than 100 thousand images per class(/label)

sequence_length = 80
train_grand= []



class_paths = glob('../input/train_simplified/*.csv')



df = []



for i,c in enumerate(tqdm(class_paths[0: num_classes])):

    train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=15000) # [2500 rows x 2 columns]

    train = train[train.recognized == True].head(10000) # use data only recognized == True -> [2000 rows x 2 columns]

    

    X = []

    for values in train.drawing.values:

        image = json.loads(values)

        strokes = []

        for x_axis, y_axis in image:

            strokes.extend(list(zip(x_axis, y_axis)))

        strokes = np.array(strokes)

        pad = np.zeros((sequence_length, 2))

        if sequence_length>strokes.shape[0]:

            pad[:strokes.shape[0],:] = strokes

        else:

            pad = strokes[:sequence_length, :]

        X.append(pad)

    X = np.array(X)

    y = np.full((train.shape[0], 1), i)

    X = np.reshape(X, (10000, -1))

    X = np.concatenate((y, X), axis=1)

    train_grand.append(X)

   

    

#     trainarray = np.reshape(trainarray, (ims_per_class, -1)) # (2000, 1024)

#     labelarray = np.full((train.shape[0], 1), i) # (2000, 1) fill with 'i' 0~339

#     trainarray = np.concatenate((labelarray, trainarray), axis=1) # (2000, 1025)

#     train_grand.append(trainarray)



train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) 

print(train_grand.shape)

train_grand = train_grand.reshape((-1, sequence_length*2+1))

print(train_grand.shape)



del X

del train
valfrac = 0.1 

cutpt = int(valfrac * train_grand.shape[0])

print(cutpt)



np.random.shuffle(train_grand)

y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]

y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:]



del train_grand



y_train = keras.utils.to_categorical(y_train, num_classes)

X_train = X_train.reshape(-1, sequence_length,2)



y_val = keras.utils.to_categorical(y_val, num_classes)

X_val = X_val.reshape(-1, sequence_length,2)



print(y_train.shape, "\n",

      X_train.shape, "\n",

      y_val.shape, "\n",

      X_val.shape)



# model = Sequential()

# model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(isequence_length,2)))

# model.add(MaxPooling2D(pool_size=(2, 2)))



# model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.2))



# model.add(Flatten())

# model.add(Dense(680, activation='relu'))

# model.add(Dropout(0.5))

# model.add(Dense(num_classes, activation='softmax'))



# model.summary()
def createNetwork(seq_len):

    

    # Function to add a convolution layer with batch normalization

    def addConv(network, features, kernel):

        network = BatchNormalization()(network)

        return Conv1D(features, kernel, padding='same', activation='relu')(network)

    

    # Function to add a dense layer with batch normalization and dropout

    def addDense(network, size):

        network = BatchNormalization()(network)

        network = Dropout(0.2)(network)

        return Dense(size, activation='relu')(network)

    

    

    # Input layer

    input = Input(shape=(seq_len, 2))

    network = input

    

    # Add 1D Convolution

    for features in [16, 24, 32]:

        network = addConv(network, features, 5)

    network = MaxPool1D(pool_size=5)(network)

    

    # Add 1D Convolution

    for features in [64, 96, 128]:

        network = addConv(network, features, 5)

    network = MaxPool1D(pool_size=5)(network)



    # Add 1D Convolution

    for features in [256, 384, 512]:

        network = addConv(network, features, 5)

    #network = MaxPool1D(pool_size=5)(network)



    # Flatten

    network = Flatten()(network)

    

    # Dense layer for combination

    for size in [128, 128]:

        network = addDense(network, size)

    

    # Output layer

    output = Dense(len(files), activation='softmax')(network)





    # Create and compile model

    model = Model(inputs = input, outputs = output)

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



#     # Display model

#     model.summary()

    return model



model = createNetwork(sequence_length)
def top_3_accuracy(x,y): 

    t3 = top_k_categorical_accuracy(x,y, 3)

    return t3



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0)



earlystop = EarlyStopping(monitor='val_loss', mode='auto', patience=2,verbose=0) 



#callbacks = [reduceLROnPlat, earlystop]

#callbacks = earlystop



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy', top_3_accuracy])



model.summary()



# model.fit(x=X_train, y=y_train,

#           batch_size = 1000,

#           epochs = 100,

#           validation_data = (X_val, y_val),

#           callbacks = callbacks,

#           verbose = 1)

model.fit(x=X_train, y=y_train,

          batch_size = 1000,

          epochs = 25,

          validation_data = (X_val, y_val),

          verbose = 1)

#%% get test set

ttvlist = []

reader = pd.read_csv('../input/test_simplified.csv', index_col=['key_id'],

    chunksize=2048)



for chunk in tqdm(reader, total=55):

    X =[]

    for values in chunk.drawing.values:

        image = json.loads(values)

        strokes = []

        for x_axis, y_axis in image:

            strokes.extend(list(zip(x_axis, y_axis)))

        strokes = np.array(strokes)

        pad = np.zeros((sequence_length, 2))

        if sequence_length>strokes.shape[0]:

            pad[:strokes.shape[0],:] = strokes

        else:

            pad = strokes[:sequence_length, :]

        X.append(pad)

        

    X = np.array(X)

    X = np.reshape(X, (-1,sequence_length, 2))

    testpreds = model.predict(X, verbose=0)

    ttvs = np.argsort(-testpreds)[:, 0:3]

    ttvlist.append(ttvs)

#     imagebag = bag.from_sequence(chunk.drawing.values).map(draw_it)

#     testarray = np.array(imagebag.compute())



#     testarray = np.reshape(testarray, (testarray.shape[0], imheight, imwidth, 1))

#     testpreds = model.predict(testarray, verbose=0)

#     ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3

#     ttvlist.append(ttvs)

    

ttvarray = np.concatenate(ttvlist)
preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

preds_df = preds_df.replace(numstonames)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



sub = pd.read_csv('../input/sample_submission.csv', index_col=['key_id'])

sub['word'] = preds_df.words.values

sub.to_csv('subcnn_small.csv')

sub.head()