import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import glob

import gc 

import os

import random

import pathlib

import cv2 as cv



tf.__version__
keras = tf.keras

layers = tf.keras.layers
rgb_dir = '/kaggle/input/hmdb51/HMDB51/'

rgb_root = pathlib.Path(rgb_dir)
def load_file(path):

    train = np.load(path)

    train = train.tolist()

    random.shuffle(train)

    train_ps=[]

    l1=[]

    l2=[]

    l3=[]

    for p in train:

        rt = pathlib.Path(p)

        ps = sorted(list(rt.glob('*.jpg')))

        num = len(ps)

        step = num//3

        for i in range(5):

            num1 = random.randint(0,step-1)

            num2 = random.randint(step,step*2-1)

            num3 = random.randint(step*2,num-1)

            l1.append(str(ps[num1]))

            l2.append(str(ps[num2]))

            l3.append(str(ps[num3]))

    l = list((l1,l2,l3))

    return l
train_ps = load_file('/kaggle/input/hmdblist/hmdb_train.npy')

test_ps = load_file('/kaggle/input/hmdblist/hmdb_test.npy')

train_count = len(train_ps[0])

test_count = len(test_ps[0])

image_count = train_count + test_count

print(image_count)
label_names = sorted(item.name for item in rgb_root.glob('*/') if item.is_dir())

label_names
label_to_index = dict((name, index) for index,name in enumerate(label_names))

label_to_index
train_labels = [label_to_index[pathlib.Path(path).parent.parent.name] for path in train_ps[0]]

test_labels = [label_to_index[pathlib.Path(path).parent.parent.name] for path in test_ps[0]]
train_label_onehot = tf.keras.utils.to_categorical(train_labels)

test_label_onehot = tf.keras.utils.to_categorical(test_labels)
def load_preprosess_image(path):

    a = [256,224,192,168]

    n1 = random.randint(0,3)

    n2 = random.randint(0,3)

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.resize(image, [256, 340])

    image = tf.image.random_contrast(image, 0.6, 1)

    image = tf.image.random_crop(image, [a[n1], a[n2], 3])

    image = tf.image.resize(image, [224, 224])

    image = tf.cast(image, tf.float32)

    image = image/255

    return image

def load_preprosess_image1(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image,[224,224])

    image = tf.cast(image, tf.float32)

    image = image/255

    return image
AUTOTUNE = tf.data.experimental.AUTOTUNE

def path2image(paths,fn):

    path_ds = tf.data.Dataset.from_tensor_slices(paths)

    image_ds = path_ds.map(fn)

    return image_ds

def image_label(paths,lot,fn):

    image_ds1 = path2image(paths[0],fn)

    image_ds2 = path2image(paths[1],fn)

    image_ds3 = path2image(paths[2],fn)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(lot, tf.int64))

    image_ds = tf.data.Dataset.zip((image_ds1, image_ds2,image_ds3))

    image_label_ds = tf.data.Dataset.zip((image_ds,label_ds))

    return image_label_ds
train_data = image_label(train_ps,train_label_onehot,load_preprosess_image)

test_data = image_label(test_ps,test_label_onehot,load_preprosess_image1)
BATCH_SIZE = 256



train_data = train_data.repeat(-1).shuffle(BATCH_SIZE*2+50).batch(BATCH_SIZE)

test_data = test_data.repeat(-1).batch(BATCH_SIZE)
train_data
covn_base = keras.applications.xception.Xception(weights='imagenet', 

                                                 include_top=False,

                                                 input_shape=(224,224, 3),

                                                 pooling='avg')

#covn_base.trainable = False

covn_base.summary()
for i in covn_base.layers:

    i.trainable = False
n_model = keras.Sequential()

n_model.add(covn_base)

n_model.summary()
input1 = keras.Input(shape=(224,224,3))

input2 = keras.Input(shape=(224,224,3))

input3 = keras.Input(shape=(224,224,3))

op1 = n_model(input1)

op2 = n_model(input2)

op3 = n_model(input3)

merged = keras.layers.Average()([op1,op2,op3])

predictions = keras.layers.Dropout(0.8)(merged)

predictions = layers.Dense(51, activation='softmax')(predictions)

model = keras.Model([input1, input2,input3], predictions)

model.summary()
# #model.layers[3].trainable =False

# for i in model.layers[3].layers[:-33]:

#     i.trainable = False

# for i in model.layers[3].layers[:-33]:

#     i.trainable = True

# model.summary()
# model.compile(optimizer=keras.optimizers.Adam(lr=0.001),

#               loss='categorical_crossentropy',

#               metrics=['acc'])
model.summary()

model.compile(optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.9),

              loss='categorical_crossentropy',

              metrics=['acc'])
cp_path = '/kaggle/working/cp_rgb_xception2.ckpt'

cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path,save_weights_only=True,sava_best_only=True)

def scheduler(epoch):

    # 每隔1个epoch，学习率减小为原来的1/10

    if  epoch != 0 and (epoch-0)%2==0:

        lr = keras.backend.get_value(model.optimizer.lr)

        keras.backend.set_value(model.optimizer.lr, lr * 0.3)

        print("lr changed to {}".format(lr * 0.3))

    return keras.backend.get_value(model.optimizer.lr)

 

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
initial_epochs = 3

history = model.fit(

    train_data,

    steps_per_epoch=(train_count//BATCH_SIZE),

    epochs=initial_epochs,

    validation_data=test_data,

    validation_steps=(test_count//BATCH_SIZE),

    callbacks=[cp_callback])
for i in covn_base.layers[-33:]:

    i.trainable = True

model.summary()

n_model = keras.Sequential()

n_model.add(covn_base)

n_model.summary()
input1 = keras.Input(shape=(224,224,3))

input2 = keras.Input(shape=(224,224,3))

input3 = keras.Input(shape=(224,224,3))

op1 = n_model(input1)

op2 = n_model(input2)

op3 = n_model(input3)

merged = keras.layers.Average()([op1,op2,op3])

predictions = keras.layers.Dropout(0.8)(merged)

predictions = model.layers[-1](predictions)

# predictions = new_model.layers[1](merged)

# predictions = new_model.layers[2](predictions)

# predictions = keras.layers.Dropout(0.9)(predictions)

# predictions = new_model.layers[4](predictions)

#predictions = layers.Dense(101, activation='softmax')(merged)

model2 = keras.Model([input1, input2,input3], predictions)

model2.summary()

model = model2
model.summary()

model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),

              loss='categorical_crossentropy',

              metrics=['acc'])
initial_epochs = 3

fine_tune_epochs = 5

total_epochs =  initial_epochs + fine_tune_epochs

history = model.fit(

    train_data,

    steps_per_epoch=(train_count//BATCH_SIZE),

    epochs=total_epochs,

    initial_epoch = initial_epochs,

    validation_data=test_data,

    validation_steps=(test_count//BATCH_SIZE),

    callbacks=[cp_callback])
model.summary()

model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),

              loss='categorical_crossentropy',

              metrics=['acc'])
initial_epochs = 8

fine_tune_epochs = 3

total_epochs =  initial_epochs + fine_tune_epochs

history = model.fit(

    train_data,

    steps_per_epoch=(train_count//BATCH_SIZE),

    epochs=total_epochs,

    initial_epoch = initial_epochs,

    validation_data=test_data,

    validation_steps=(test_count//BATCH_SIZE),

    callbacks=[cp_callback])
covn_base.save('/kaggle/working/hmdb_model.h5')
train_data = image_label(train_ps,train_label_onehot,load_preprosess_image1)

test_data = image_label(test_ps,test_label_onehot,load_preprosess_image1)

BATCH_SIZE = 256



train_data = train_data.repeat(-1).shuffle(BATCH_SIZE*2+50).batch(BATCH_SIZE)

test_data = test_data.repeat(-1).batch(BATCH_SIZE)
initial_epochs = 11

fine_tune_epochs = 4

total_epochs =  initial_epochs + fine_tune_epochs

history = model.fit(

    train_data,

    steps_per_epoch=(train_count//BATCH_SIZE),

    epochs=total_epochs,

    initial_epoch = initial_epochs,

    validation_data=test_data,

    validation_steps=(test_count//BATCH_SIZE),

    callbacks=[cp_callback])
covn_base.save('/kaggle/working/hmdb_model2.h5')