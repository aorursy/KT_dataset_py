import numpy as np

import pandas as pd

import os

import random, re, math

import tensorflow as tf, tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers

from kaggle_datasets import KaggleDatasets



print(tf.__version__)

print(tf.keras.__version__)
!pip install efficientnet

import efficientnet.tfkeras as efn
# tf.data.experimental.AUTOTUNE??
AUTO = tf.data.experimental.AUTOTUNE #basic Convert a number or string to an integer, or return 0 if no arguments

#are given. for more detail uncommen the cell above

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)





# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
from matplotlib import pyplot as plt



img = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg')

print(img.shape)

plt.imshow(img)
path='../input/plant-pathology-2020-fgvc7/'

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

sub = pd.read_csv(path + 'sample_submission.csv')



train_paths = train.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values #put out datapath on the TPU

test_paths = test.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values #put out datapath on the TPU



train_labels = train.loc[:, 'healthy':].values #you can also use '1' instead of 'healthy' but all it does is taking all the results the 4 different labels
train #Seeing the training csv-file
train_labels #all the rows for the labels we want to classify
from pathlib import Path

import PIL,os,mimetypes

Path.ls = lambda x: list(x.iterdir())
path_images = Path('/kaggle/input/plant-pathology-2020-fgvc7')

path_images.ls()
(path_images/'images').ls()[:3]
nb_classes = 4 #number of labels, this will be used for our output layer

BATCH_SIZE = 8 * strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU #try change it to 16

img_size = 768 #u decide but bigger images take longer to train but higher kvali and smaller images is faster but lower kvali

EPOCHS = 25 #number of training rounds
x = tf.constant([1.8, 2.2], dtype=tf.float32)

print(type(x))

tf.dtypes.cast(x, tf.int32)  # [1, 2], dtype=tf.int32

print(type(x))
type(x)
#decode_image label every image if it has a label from csv-file and return also the image that has no label

def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)#get the filename

    image = tf.image.decode_jpeg(bits, channels=3) #channel representing some aspect of information about the image 

    #and u can have 100+ channels if you want but it is proven to be good to put it at 3

    image = tf.cast(image, tf.float32) / 255.0 #tf.cast means = Casts a tensor to a new type. so the images tensor becomes a type float

    image = tf.image.resize(image, image_size) #resize the image for the size given above

    if label is None:  #if there is no label for an image

        return image #jusst return the image

    else:

        return image, label #else return the image with the label

    

#data_augment just take the images and do som augment to them

def data_augment(image, label=None, seed=2020): 

    image = tf.image.random_flip_left_right(image, seed=seed) #flip randomly images to left or right

    image = tf.image.random_flip_up_down(image, seed=seed) #flip the images randomly up or down

    image=tf.image.adjust_saturation(image, 2)

#     image=tf.image.resize_with_crop_or_pad(img, 800, 900)

    

    #for every new image

    if label is None:  #if there is no label for an image

        return image#jusst return the image

    else:

        return image, label #else return the image with the label
saturated = tf.image.adjust_saturation(img, 10)

plt.imshow(saturated)
plt.imshow(img) #if we look at the before and after picture it is clear to see that illness of the plant is much clear on the saturated image, so we are gonna use it in our data aurgment function above 
#making the training dataset for more detail on the given functions uncommen the cell below

train_dataset = (

    tf.data.Dataset #explaned above nut just a API blok iniziator

    .from_tensor_slices((train_paths, train_labels)) #Creates a `Dataset` whose elements are slices of the given tensors

    # 'from_tensor_slices' --> The given tensors are sliced along their first dimension. This operation

#     preserves the structure of the input tensors, removing the first dimension

#     of each tensor and using it as the dataset dimension. All input tensors

#     must have the same size in their first dimensions.

    .map(decode_image, num_parallel_calls=AUTO) #Maps `map_func` across the elements of this dataset# note here is 'map_func'='decode_image' which returned the labels with the images

    #This transformation applies `map_func` to each element of this dataset, and

#     returns a new dataset containing the transformed elements, in the same

#     order as they appeared in the input. `map_func` can be used to change both

#     the values and the structure of a dataset's elements. For example, adding 1

#     to each element, or projecting a subset of element components.

    .map(data_augment, num_parallel_calls=AUTO)#note here we use data_augment wich fliped the images for ech element in the dataset

    .repeat() #Repeats this dataset so each original value is seen `count` times. #see exsampel below #  The default behavior (if

        #count` is `None` or `-1`) is for the dataset be repeated indefinitely.



    .shuffle(512) #Randomly shuffles the elements of this dataset.#and it randomize by 512 (u set value) each time

    .batch(BATCH_SIZE) 

    .prefetch(AUTO) #Creates a `Dataset` that prefetches elements from this dataset.

#     Most dataset input pipelines should end with a call to `prefetch`. This

#     allows later elements to be prepared while the current element is being

#     processed. This often improves latency and throughput, at the cost of

#     using additional memory to store prefetched elements.

    )
#  tf.data.Dataset??
test_dataset = (

    tf.data.Dataset #explaned above nut just a API blok iniziator

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=AUTO) 

    #note we dont flip the testing images because then we cant validate the predictions from the model

    .batch(BATCH_SIZE)

)
# tf.keras.callbacks??
#predifened learning rates for optimal preformaze

LR_START = 0.00001

LR_MAX = 0.0001 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 15

LR_SUSTAIN_EPOCHS = 3

LR_EXP_DECAY = .8

#here we change the learning depending on what epoch we are at 

def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS: #if epoch is lower then 15

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START #take this learning rate

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:#if epoch is lower then 15+3=18 

        lr = LR_MAX #then take this learning rate

    else: #else if is above 18 then 

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN#take this learning rate

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True) #create a callback to make the functionality possible when we do the actual training. 

#verbose = true just means we want to see the traning proces bar



#plot the learning rate schedular 

rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def get_model():

    #pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)

#     pretrained_model = tf.keras.applications.Xception(weights='imagenet', input_shape=(img_size, img_size, 3), include_top=False)

    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    #pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    # EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)

    pretrained_model = efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))

    pretrained_model.trainable = True

    

    model = tf.keras.Sequential([

        pretrained_model,

#         tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(4096, activation='relu'),

        tf.keras.layers.Dense(4096, activation='relu'),

        tf.keras.layers.Dense(4, activation='softmax') #4 since there are 4 labels and therefor 4 diferent predictions 

    ])



#     x = pretrained_model.output

#     predictions = Dense(4, activation="softmax")(x)

    return model

with strategy.scope():

    model = get_model()

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
%%time

history=model.fit(

    train_dataset, 

    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,

    callbacks=[lr_callback], #lr_callback was created above

    epochs=EPOCHS

)
%%time

probs = model.predict(test_dataset)
sub.loc[:, 'healthy':] = probs #u can also use 1 instead of 'healthy'

sub.to_csv('submission.csv', index=False)

sub.head()