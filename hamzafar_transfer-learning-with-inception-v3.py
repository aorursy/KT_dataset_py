import pymongo

from pymongo import MongoClient

client = MongoClient()

client = MongoClient('localhost', 27017)

# client = MongoClient('mongodb://localhost:27017/')

db = client.db_cdiscount
import io

import bson                       # this is installed with the pymongo package

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data
import numpy as np

import scipy

import time
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras import backend as K
# create the base pre-trained model

base_model = InceptionV3(weights='imagenet', include_top=False)
first = 0

last = 10

batch = 4

num_batch = int(last/batch)

# last = 82
epochs = 10

batch_size = 4

num_classes = db.cat_encode.count()
x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 9 class

predictions = Dense(num_classes, activation='softmax')(x)
# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)
# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
start_time = time.time()

print('start code')

counter = 0

row_count = 0

for j in range(0,num_batch-1):

# for j in range(0,1):

    batch_time = time.time()

    a1 = int(list(np.linspace(first,last,num_batch))[j])

    an = int(list(np.linspace(first,last,num_batch))[j+1])

    

    lst_batch = []

    

    print(a1,an)

    

    cur = db.train.find({})[a1:an]

    

    m= an-a1

        

    i = 0

#     pic_array

    while (cur.alive):

        idx = cur.next()

        dic = {}

        

        category_id =idx['category_id'] 

    #     print(category_id)



        cat = db.cat_encode.find_one({ "cat" : (category_id)}, {"cat" : 1.0, "_id" : 0})['cat']

    #     print(cat)

        

        picture = (imread(io.BytesIO(idx['imgs'][0]['picture'])))

        picture  = np.float32(scipy.misc.imresize(picture, (150,150), interp='bilinear', mode=None)/255.0)        

        dic['picture'] = picture

        

        encode = db.cat_encode.find_one({ "cat" : (category_id)}, {"encode" : 1.0, "_id" : 0})['encode']

    #     print(encode)

        dic['encode'] = encode

        

        lst_batch.append(dic)

        

        i+=1

    X_batch = np.array([lst['picture'] for lst in lst_batch])

    Y_batch = np.array([lst['encode'] for lst in lst_batch])

    print('X batch size: ', X_batch.shape)

    print('Y batch size: ', Y_batch.shape)

    

    del(lst_batch)

        

    model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, validation_split=0.3)

    

#     print(X_batch.shape[0])

    row_count = row_count + X_batch.shape[0]

    counter = counter + X_batch.shape[0]

    

    if(row_count >= 100000):

        print('row count: ',row_count)

        counter = counter + row_count

        model_name = 'model_' + str(counter) +'.h5'

        print(model_name)

#         model.model.save('E://kaggle//Cdiscount//model//'+model_name)

        print('Model saved')

        row_count = 0

    

    print("--- %s seconds ---" % (time.time() - batch_time))

    print('rows completed: ', counter)

    

print("--- %s seconds ---" % (time.time() - start_time))

# model.model.save('E://kaggle//Cdiscount//model//final_model.h5')