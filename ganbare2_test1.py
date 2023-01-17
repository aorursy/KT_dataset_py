import keras

import os

import pandas as pd

import tensorflow as tf

import numpy as np

from collections import defaultdict

from keras.models import Model

from keras.applications import InceptionV3

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Lambda

from keras.optimizers import SGD

from keras import backend as K

from keras.callbacks import ModelCheckpoint

import random

import cv2

# from ../input/notebooks/loss_functions import *

%run ../input/notebooks/loss_functions.py

# from loss_functions import *

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from collections import defaultdict

import os

from sklearn import preprocessing



# Any results you write to the current directory are saved as output.
class DataGenerator(keras.utils.Sequence):

    #'Generates data for Keras'

    def __init__(self, dictionary , classes,labels ,class_per_batch=50, batch_size=4, dim=(96,96), n_channels=3,shuffle=True):

        # 'Initialization'

        self.dim = dim

        self.dictionary = dictionary

        self.batch_size = batch_size

        self.class_per_batch = class_per_batch

        self.labels = labels

        self.n_channels = n_channels

        self.classes = classes

        self.shuffle = shuffle

        self.on_epoch_end()

        

    def __getitem__(self, index):

        #'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.class_per_batch:(index+1)*self.class_per_batch]

        # Find list of IDs

        lab = [self.labels[k] for k in indexes]

        # Generate data

        X, y = self.__data_generation(lab)

        return X, y

    

    

    def __len__(self):

        #'Denotes the number of batches per epoch'

        return int(np.floor(len(self.labels) / (self.class_per_batch)))        



    def __data_generation(self, lab):

        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((self.batch_size*self.class_per_batch, *self.dim, self.n_channels))

        y = np.empty((self.batch_size*self.class_per_batch))



        # Generate data

        #print(lab)

        #print(len(lab))

        for ID in lab:

            # Store sample

            X_temp = random.sample(self.dictionary[ID], 4)

            #print(X_temp)

            for i, sample in enumerate(X_temp):

                #print('\n\n\n\n\n',sample)

                #print(ID)

                img = cv2.imread('../input/dataset/train/train/'+sample)

                sample = cv2.resize(img  , self.dim)

                X[i,] = sample

                y[i] = ID



        return X, y

    

    def on_epoch_end(self):

        #'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.labels))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)   
# import os

# os.chdir("../input/incept/")

# from rgg import *

print(os.getcwd())
INCEPTIONV3 = InceptionV3(weights='../input/inceptweight/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

# INPUT_HEIGHT = 224

# INPUT_WIDTH = 224

# INPUT_MEAN = 127.5

# INPUT_STD = 127.5

checkpoint = ModelCheckpoint('weights-best1.hdf5',monitor = "val_loss", verbose = 1,

  save_best_only = False, save_weights_only = False, mode = "auto",

  period = 1)
df = pd.read_csv('../input/preprocessed/file_name.csv')

df = df.sort_values(by=['Id'])

len(df)

traindf = df[:20000]

valdf = df[20000:]
# labels = traindf['Id'].unique().tolist()

le = preprocessing.LabelEncoder()

classes = df['Id'].tolist()

le.fit(classes)



classes = le.transform(classes) 

# print(type(classes),classes)

labels = np.unique(classes).tolist()

# print(labels)

images = df['Image'].tolist()

dictionary = defaultdict(list)

for label, image in zip(classes, images):

    dictionary[label].append(image)
def model(original_model):

    

    bottleneck_input  = original_model.get_layer(index=0).input

    bottleneck_output = original_model.get_layer(index=-2).output

    bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)



    for layer in bottleneck_model.layers:

        layer.trainable = False

        

    new_model = Sequential()

    new_model.add(bottleneck_model)



    new_model.add(Dense(1024,input_dim  = 2048,activation = 'relu'))

    new_model.add(Dense(512,activation = 'relu'))

    new_model.add(Dense(128,activation = 'relu'))

    new_model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    print(new_model.summary())

    new_model.save('model.h5')



    return new_model
training_generator = DataGenerator(dictionary,classes, labels)

# validation_generator = DataGenerator(dictionary,classes, labels)
training_model = model(INCEPTIONV3)




# import os

# os.chdir("../input/notebooks/")

# from loss_function import *



# training_model.load_weights('../input/inceptweight/weightsbest.hdf5')
callbacks_list = [checkpoint]

print(os.getcwd())



training_model.compile(optimizer = 'adam',loss = we_loss ,metrics= [accuracy])

training_model.fit_generator(generator=training_generator,epochs=100 ,callbacks=callbacks_list,verbose=1)

training_model.save_weights('weights1.hdf5')
callbacks_list = [checkpoint]

training_model.load_weights('weights-best.hdf5')



training_model.compile(optimizer = 'adam',loss = we_loss ,metrics= [accuracy])
newdict = defaultdict(list)

count = 1

for label, image in zip(classes, images):

    

    img = cv2.imread('train/'+image)

    sample = cv2.resize(img  , (96,96))

    g = training_model.predict(np.array([sample]))

    newdict[label].append(g)

    count+=1

    print("processed :" ,count,end='\r')
# np.save('my_file.npy', newdict) 

read_dictionary = np.load('my_file.npy').item()

emmbed = defaultdict(list)

count =0 

for k in read_dictionary.keys():

    z = np.array(read_dictionary[k])

    a = z.shape[0]

    

    z  = z.reshape(a,128)

    z = z.mean(0)

    emmbed[k].append(z)

    count+=1

    print("processed :", count)

    



    

np.save('emmbedings.npy', emmbed) 

# with open('filename.pickle', 'wb') as handle:

#     pickle.dump(newdict, handle, protocol=pickle.HIGHEST_PROTOCOL)



# with open('filename.pickle', 'rb') as handle:

#     b = pickle.load(handle)
read_dictionary = np.load('emmbedings.npy').item()

from os import listdir

imglist = listdir('test')

import csv 

for d in imglist:

    li = []

    img = cv2.imread('test/'+d)

    sample = cv2.resize(img  , (96,96))

    g = training_model.predict(np.array([sample]))

    for k in read_dictionary.keys():

        dist = float(np.linalg.norm(g[0] - read_dictionary[k][0]))

        li.append([dist,k])

    li.sort()

#     print(li[0])

    sd = le.inverse_transform([li[0][1],li[1][1],li[2][1],li[3][1]])

#     print(sd)

    str1 = 'new_whale '+sd[0]+' '+sd[1]+' '+sd[2]+' '+sd[3]

    fields=[d,str1]

    with open(r'ans.csv', 'a') as f:

        writer = csv.writer(f)

        writer.writerow(fields)

        