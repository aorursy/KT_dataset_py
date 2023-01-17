#additional imports

import PIL.Image, PIL.ImageFont, PIL.ImageDraw

from matplotlib import pyplot as plt

import os, re, time, json

from keras import layers,models

from keras.layers.core import Dense

from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.regularizers import l2
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Detect hardware, return appropriate distribution strategy

import tensorflow as tf

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None

# select the appropriate distribution strategy

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("Number of accelerators: ", strategy.num_replicas_in_sync)
# important parameters

BATCH_SIZE = 32 * strategy.num_replicas_in_sync # the global batchsize

#load the data from csv to a pandas dataframe

train_df = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv") 

print(train_df)

train_df.shape
#prepare the target-values an call it 'labels'

y_train_df = train_df['label']#seperate the label-colummn as the target-column y_train_df

training_labels = y_train_df

x_train_df = train_df.drop(labels = ['label'],axis = 1)

# call the x_train_df "images" an scale it from [0..255] to [0..1]

training_images = x_train_df/255

training_images = training_images.values.astype(np.float32)
# create the train- and validation-data for training only

from sklearn.model_selection import train_test_split

training_images,val_images,training_labels,val_labels = train_test_split(training_images, training_labels, test_size=0.1, random_state=None )

print(training_images.shape)

print(training_labels.shape)

print(val_images.shape)

print(val_labels.shape)
# like to see the first training_image

myfirstdigit = training_images[0]

#print(myfirstdigit)

plt.imshow(myfirstdigit.reshape(28,28),cmap=plt.cm.binary)

plt.show()
# like to see the last training image

myfirstdigit = training_images[4860]

#print(myfirstdigit)

plt.imshow(myfirstdigit.reshape(28,28),cmap=plt.cm.binary)

plt.show()
# the validation data "Dig-Mnist" are for the evaluation

final_val_set_df = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv") 

print(final_val_set_df)

#prepare the validation-target-values and call it 'labels'

final_y_val_df = final_val_set_df['label']#seperate the label-cloumn as the target-column y_train_df

final_val_labels = final_y_val_df

final_x_val_df = final_val_set_df.drop(labels = ['label'],axis = 1)

# call the x_train_df "images" an scale it from [0..255] to [0..1]

final_val_images = final_x_val_df/255

final_val_images = final_val_images.values.astype(np.float32)
submission_df = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv') # "dtype=str" is importend for the later flow_from_dataframe-method

submission_df.dtypes
# build two input pipeline: one for the train images,labels and one for the validation images,labels



def get_training_dataset(the_images,the_labels,batch_size):

    # convert the training-data (images,labels) to a tf.dataset

    dataset = tf.data.Dataset.from_tensor_slices((the_images,the_labels))

    # shuffle, repeat and batch the samples

    dataset = dataset.cache()# a small dataset can be cached in RAM

    dataset = dataset.shuffle(1000, reshuffle_each_iteration = True)

    dataset = dataset.repeat()# this ist manadatory for keras at this point

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(-1)# fetch new batches while training on the current one

    # return the dataset

    return dataset



def get_validation_dataset(the_images,the_labels,batch_size):

    # convert the training-data (images,labels) to a tf.dataset

    dataset = tf.data.Dataset.from_tensor_slices((the_images,the_labels))

    # shuffle, repeat and batch the samples

    dataset = dataset.cache()# a small dataset can be cached in RAM

    dataset.shuffle(1000, reshuffle_each_iteration = True)

    dataset = dataset.repeat()# this ist manadatory for keras at this point

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(-1)# fetch new batches while training on the current one

    # return the dataset

    return dataset





def get_final_validation_dataset(the_images,the_labels,batch_size):

    # convert the training-data (images,labels) to a tf.dataset

    dataset = tf.data.Dataset.from_tensor_slices((the_images,the_labels))

    # shuffle, repeat and batch the samples

    dataset = dataset.cache()# a small dataset can be cached in RAM

    dataset.shuffle(1000, reshuffle_each_iteration = True)

    dataset = dataset.repeat()# this ist manadatory for keras at this point

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(-1)# fetch new batches while training on the current one

    # return the dataset

    return dataset



# After I had problems with the computing of the private score by submission I propagate

# the test_images direktly without making a dataset from them.

# Fortunately, it works.



#def get_test_dataset(the_images,batch_size):# only images, no labels => for prediction

    # convert the training-data (images,labels) to a tf.dataset

    #dataset = tf.data.Dataset.from_tensor_slices((the_images))

    # shuffle, repeat and batch the samples

    #dataset = dataset.cache()# a small dataset can be cached in RAM

    #dataset = dataset.repeat()# this ist manadatory for keras at this point

    #dataset = dataset.batch(batch_size, drop_remainder=True)

    #dataset = dataset.prefetch(-1)# fetch new batches while training on the current one

    # return the dataset

    #return dataset

# instantiate the datasets now

training_dataset = get_training_dataset(training_images,training_labels,BATCH_SIZE)

val_dataset=get_validation_dataset(val_images,val_labels,BATCH_SIZE)

final_validation_dataset = get_final_validation_dataset(final_val_images,final_val_labels,BATCH_SIZE)



# test_dataset = get_test_dataset(test_images,len(test_images))#all 5000 images # no need for prediction

# Activation funktion "Leaky ReLU" is one attempt to fix the “dying ReLU” problem. 

# Instead of the function being zero when x < 0, a leaky ReLU will instead have a 

# small negative slope (of 0,1, 0.01, or so).

# In this model I use a parametrical LeakyRelu with paramater alpha: f(alpha,x)=alpha*x for x<0, =x for x>=0





def make_my_model ():

    model = tf.keras.Sequential(

    [tf.keras.layers.Reshape(input_shape=(28*28,),target_shape=(28,28,1),name="image"),

    

    tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same'), # no bias necessary before batch norm

    tf.keras.layers.BatchNormalization(), # no batch norm scaling necessary before "relu"

    tf.keras.layers.ReLU(), 

     

    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'), 

    tf.keras.layers.BatchNormalization(), 

    tf.keras.layers.ReLU(), 

     

    tf.keras.layers.MaxPool2D((2,2)), 

    tf.keras.layers.Dropout(0.2),

     

    tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), # in the inner layer, my results are better with LeakyRelu

    

    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1),  

     

    tf.keras.layers.MaxPool2D((2,2)),

    tf.keras.layers.Dropout(0.2), 

    

    tf.keras.layers.Flatten(),

     

    tf.keras.layers.Dense(256),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), 

        

    tf.keras.layers.Dense(128),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), 

     

    tf.keras.layers.Dropout(0.2),

     

    tf.keras.layers.Dense(32),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LeakyReLU(alpha=0.1), 

    

    tf.keras.layers.Dense(10, activation='softmax')# the last layer for the classes 0..9

       

    ])

    

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

with strategy.scope():

    model = make_my_model()



model.summary()
# Training parameters

EPOCHS = 100

STEPS_PER_EPOCH = 5400//BATCH_SIZE # 60,000 items in this dataset, 54000 train/6000 val

print("Steps per epoch: ", STEPS_PER_EPOCH)



# parameters

#BATCH_SIZE = 64 * strategy.num_replicas_in_sync # the global batchsize



LEARNING_RATE =0.01

if (strategy.num_replicas_in_sync == 1):

    LEARNING_RATE_EXP_DECAY = 0.6

else: LEARNING_RATE_EXP_DECAY = 0.7



LEARNING_RATE_DECAY = tf.keras.callbacks.LearningRateScheduler(

                      lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY ** epoch,

                    verbose=0)



history = model.fit(training_dataset,

                    steps_per_epoch=STEPS_PER_EPOCH,

                    epochs=EPOCHS,

                    callbacks=[LEARNING_RATE_DECAY],

                    validation_data=(val_dataset),

                    validation_steps=STEPS_PER_EPOCH,

                    verbose=2

                   )



# evaluate the model by using the "Dig-MNIST.csv"-data set for separate evaluation

final_stats = model.evaluate(final_validation_dataset, steps=1)

print("Accuracy of the validation-set: ", final_stats[1])
history_dict = history.history

history_dict.keys()# look witch keys exist

loss_values = history_dict['loss']

acc_values = history_dict['accuracy']

epochs = range(1, len(loss_values)+1)

plt.plot(epochs,loss_values,'r', label='training loss')

plt.plot(epochs,acc_values,'g',label='accuracy')

plt.title('loss and accuracy')

plt.xlabel('epochs')

plt.ylabel('loss vs. accuracy')

plt.legend()

plt.show()
#load the data from csv to a pandas df

test_df = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

print(test_df) # the test-data-set contains an "id"-column

test_df.shape
x_test_df = test_df.drop(labels = ['id'],axis = 1)# drop the id-column for the test-dataset

print(x_test_df)

test_images = x_test_df/255

test_images = test_images.values.astype(np.float32)

test_images.shape
results = model.predict_classes(test_images)# I propagate the test_images directly to avoid erros by compute the private score

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = results # entry the predicted result into the 'label' column

submission

submission.to_csv("submission.csv", index=False)