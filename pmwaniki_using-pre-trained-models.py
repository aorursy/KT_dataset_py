# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



tf.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#get data

train_paths=[]

train_labels=[]

for root,dir,files in os.walk("../input/train"):

    for file in files:

        train_paths.append(os.path.join(root,file))

        train_labels.append(root.split("/")[-1])



test_paths=[]

for root,dir,files in os.walk("../input/test"):

    for file in files:

        test_paths.append(os.path.join(root,file))

        

print(train_paths[:5])

print(test_paths[:5])
#function to preprocess images and labels

def preprocess_image(image,image_shape=[192,192]):

  image = tf.image.decode_jpeg(image, channels=3)

  image = tf.image.resize(image, image_shape)

  image /= 255.0  # normalize to [0,1] range

  return image





def load_and_preprocess_image(file,image_shape):

    image = tf.io.read_file(file)

    return preprocess_image(image,image_shape)



labs2index={'cbb':0, 'cbsd':1, 'cgm':2, 'cmd':3, 'healthy':4}

index2labs={0:'cbb', 1:'cbsd', 2:'cgm', 3:'cmd', 4:'healthy'}

def process_labels(lab):

    print(lab)

    return tf.one_hot(lab,depth=len(labs2index))
#create tensorflow data objects

train_ds = tf.data.Dataset.from_tensor_slices(train_paths)

test_ds=tf.data.Dataset.from_tensor_slices(test_paths)

train_image_ds = train_ds.map(lambda x:load_and_preprocess_image(x,image_shape=[192,192]),num_parallel_calls=AUTOTUNE)

test_image_ds=test_ds.map(lambda x:load_and_preprocess_image(x,image_shape=[192,192]),num_parallel_calls=AUTOTUNE)



train_label_ds=tf.data.Dataset.from_tensor_slices([labs2index[i] for i in train_labels]).map(process_labels,num_parallel_calls=AUTOTUNE)

image_label_ds=tf.data.Dataset.zip((train_image_ds,train_label_ds))



BATCH_SIZE = 32



# Setting a shuffle buffer size as large as the dataset ensures that the data is

# completely shuffled.

ds = image_label_ds.shuffle(buffer_size=len(train_paths))

ds = ds.repeat()

ds = ds.batch(BATCH_SIZE)

# `prefetch` lets the dataset fetch batches, in the background while the model is training.

ds = ds.prefetch(buffer_size=AUTOTUNE)



ds_test=test_image_ds.batch(BATCH_SIZE)

print(ds)

print(test_ds)
#use pretrained model

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)

#mobile_net.trainable=False



def change_range(image,label):

  return 2*image-1, label



keras_ds = ds.map(change_range)

keras_test=ds_test.map(lambda x:2*x-1)





#image_batch, label_batch = next(iter(keras_ds))



model = tf.keras.Sequential([

  mobile_net,

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(len(labs2index),activation=tf.nn.softmax)])



model.compile(optimizer=tf.train.AdamOptimizer(),

              loss=tf.keras.losses.categorical_crossentropy,

              metrics=["accuracy"])

print(model.summary())
#fit model for 50 epochs

model.fit(keras_ds, epochs=50, steps_per_epoch=176)



#make predictions

test_predictions=model.predict(keras_test,steps=int(np.ceil(len(test_paths)/BATCH_SIZE)))

predictions=np.argmax(test_predictions,axis=1)
#make submission

my_submission = pd.DataFrame({'Category':[index2labs[j] for j in predictions],'Id':[i.split("/").pop() for i in test_paths]})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)

print(my_submission.head())