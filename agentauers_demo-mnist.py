import numpy as np

import pandas as pd

import os

import random, re, math, time



from os.path import join 



import tensorflow as tf

import tensorflow.keras.backend as K

#import tensorflow_addons as tfa



from tqdm.keras import TqdmCallback



from PIL import Image



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold



from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report







import plotly

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



random.seed(a=42)

AUTO = tf.data.experimental.AUTOTUNE

strategy = tf.distribute.get_strategy()
BASEPATH = "../input/digit-recognizer"

CLASSES = 10

SIZE = 28

CNT_TRAIN = 42000

CNT_TEST = 28000



df = pd.read_csv(join(BASEPATH, 'train.csv')).values.astype(np.uint8)

x_train_all = df[:,1:].reshape((CNT_TRAIN, SIZE, SIZE))

y_train_all = df[:,0]



df = pd.read_csv(join(BASEPATH, 'test.csv')).values.astype(np.uint8)

x_test = df.reshape((CNT_TEST, SIZE, SIZE))
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))



def transform(image):

    

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = SIZE*2

    XDIM = DIM%2 #fix for size 331

    

    rot = 15.0 * tf.random.normal([1],dtype='float32')

    shr = 12.0 * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/8.0

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/8.0

    h_shift = 4.0 * tf.random.normal([1],dtype='float32') 

    w_shift = 4.0 * tf.random.normal([1],dtype='float32') 



    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM])
def prepare_image(img, label=None, augment=True):

    img = tf.cast(img, dtype=tf.float32)

    img = img / 255

    

    img = tf.reshape(img, (SIZE, SIZE, 1))

    img = tf.image.resize(img, (SIZE*2,SIZE*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)

            

    if augment:

        if 0.9 > tf.random.uniform([1], minval=0, maxval=1):

            img = transform(img)

            

    img = tf.reshape(img, (SIZE*2, SIZE*2, 1))

    img = tf.image.resize(img, (SIZE,SIZE), method=tf.image.ResizeMethod.BILINEAR, antialias=True)

    img = tf.reshape(img, (SIZE, SIZE))

    

    if label is None:

        return img

    else:

        #label = tf.cast(label, dtype=tf.float32)

        label = tf.one_hot(label, CLASSES)

        return img, label
def getTrainDataset(x, y, augment = True, shuffle = True):

    ds = tf.data.Dataset.from_tensor_slices((x, y)) 

    ds = ds.repeat()

    

    if shuffle:

        ds = ds.shuffle(1024)

        

    

    ds = ds.map(lambda img, label: prepare_image(img, label, augment=augment), num_parallel_calls=AUTO)

    ds = ds.batch(32*strategy.num_replicas_in_sync)

    ds = ds.prefetch(AUTO)

    return ds
it = iter(getTrainDataset(x_train_all, y_train_all).unbatch())



img = Image.new(mode='L', size=(32*SIZE+31, 10*SIZE+9), color=30)

for i in range(32*10):    

    (img2, label) = next(it)

    img2 = img2*255

    ix = i % 32

    iy= i // 32

    img1 = Image.fromarray(img2.numpy().reshape((SIZE,SIZE)).astype(np.uint8))

    img.paste(img1, (ix*SIZE+ix, iy*SIZE+iy))

        

display(img)

def getTestDataset(x, augment = False, repeat = False):

    ds = tf.data.Dataset.from_tensor_slices((x)) 

    

    if repeat:

        ds = ds.repeat()     

        

    ds = ds.map(lambda img: prepare_image(img, augment=augment), num_parallel_calls=AUTO)

    ds = ds.batch(32*strategy.num_replicas_in_sync)

    ds = ds.prefetch(AUTO)

    return ds

it = iter(getTestDataset(x_test).unbatch())



img = Image.new(mode='L', size=(32*SIZE+31, 10*SIZE+9), color=30)

for i in range(32*10):    

    img2 = next(it)

    img2 = img2*255

    ix = i % 32

    iy= i // 32

    img1 = Image.fromarray(img2.numpy().reshape((SIZE,SIZE)).astype(np.uint8))

    img.paste(img1, (ix*SIZE+ix, iy*SIZE+iy))

        

display(img)

def get_model():

    cur = tf.keras.Input(shape=(SIZE, SIZE), name="imgIn")

    model_input = cur

    

    cur = tf.keras.layers.Reshape((SIZE, SIZE, 1))(cur)



    cur = tf.keras.layers.Conv2D(16, kernel_size=(7, 7), activation="relu")(cur)

    cur = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cur)

    

    cur = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(cur)

    cur = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cur)

    

    cur = tf.keras.layers.Flatten()(cur)

    #cur = tf.keras.layers.Dropout(0.5)(cur)

    cur = tf.keras.layers.Dense(CLASSES, activation="softmax")(cur)      

    

    model = tf.keras.Model(model_input, cur, name='aNetwork')

    

    return model

def get_model_2():

    cur = tf.keras.Input(shape=(SIZE, SIZE), name="imgIn")

    model_input = cur

    

    cur = tf.keras.layers.Reshape((SIZE, SIZE, 1))(cur)



    cur = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation="relu")(cur)

    cur = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cur)

    

    cur = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(cur)

    cur = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cur)

    

    cur = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(cur)

    cur = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cur)

    

    cur = tf.keras.layers.Flatten()(cur)

    #cur = tf.keras.layers.Dropout(0.5)(cur)

    cur = tf.keras.layers.Dense(CLASSES, activation="softmax")(cur)      

    

    model = tf.keras.Model(model_input, cur, name='aNetwork')

    

    return model
strategy = tf.distribute.get_strategy()



with strategy.scope():

    model = get_model()

 

    model.compile(

        optimizer = 'adam',

        #loss = tf.keras.losses.SparseCategoricalCrossentropy(),

        loss = tf.keras.losses.CategoricalCrossentropy(),

        metrics=['categorical_accuracy']

    )
model.summary()
def visualizeCnnFilters(w):

    w = w.numpy()

    (tx, ty, bits, cnt) = w.shape

    w = w/w.std()

    w = w/2

    np.clip(w, -1, 1, out=w)

    w = (w*127+127).astype(np.uint8)

    tilesy = (cnt-1) // 10

    img = Image.new(mode='L', size=(tx*10+9, ty*(tilesy+1)+tilesy))

    

    for i in range(cnt):

        ix = i % 10

        iy = i // 10

        img2 = Image.fromarray(w[:,:,0,i])

        img.paste(img2, (ix*tx+ix, iy*ty+iy))

     

    return img
display(visualizeCnnFilters(model.weights[0]))
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.33, random_state=42)

ds_train = getTrainDataset(x_train, y_train, augment = True)

ds_val = getTrainDataset(x_val, y_val, augment = False)



stepsTrain = len(y_train) / (32 * strategy.num_replicas_in_sync)

stepsVal   = len(y_val) / (32 * strategy.num_replicas_in_sync)
history = model.fit(ds_train, 

                    validation_data=ds_val, 

                    verbose = 1,

                    steps_per_epoch=stepsTrain, 

                    validation_steps=stepsVal, 

                    epochs=5)

display(visualizeCnnFilters(model.weights[0]))
ds_test = getTestDataset(x_test)

probs_test = model.predict(ds_test, verbose=1)

y_test = np.argmax(probs_test, axis=1)

submission = pd.read_csv(join(BASEPATH, 'sample_submission.csv'))

submission['Label'] = y_test

submission.to_csv('submission.csv', index = False)

!head submission.csv