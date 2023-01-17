# Remove warning messages

import warnings

warnings.filterwarnings('ignore')



import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import plotly

import plotly.graph_objects as go

%matplotlib inline



import os



import math

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import tensorflow as tf

from keras.utils.np_utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.metrics import top_k_categorical_accuracy, categorical_accuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
# Set seed

np.random.seed(42)
print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
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
PATH_TO_DATA = '../input/digit-recognizer/' # This is where the data is stored on this notebook
# Load train and test

train = pd.read_csv(PATH_TO_DATA + 'train.csv')

test = pd.read_csv(PATH_TO_DATA + 'test.csv')
def preprocessing(train, test, split_train_size = 0.1):



    X_submission = train.drop(["label"],axis = 1)

    y_submission = train["label"]



    # Normalize the data

    X_submission = X_submission / 255.0

    test = test / 255.0



    # Reshape into right format vectors

    X_submission = X_submission.values.reshape(-1,28,28,1)

    X_test = test.values.reshape(-1,28,28,1)



    # Apply ohe on labels

    y_submission = to_categorical(y_submission, num_classes = 10)

    

    # Split the train and the validation set for the fitting

    X_train, X_val, y_train, y_val = train_test_split(X_submission, y_submission, test_size = split_train_size, random_state=42)

    

    return X_train, y_train, X_val, y_val, X_test, X_submission, y_submission



X_train, y_train, X_val, y_val, X_test, X_submission, y_submission = preprocessing(train, test)
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)

print(X_test.shape)

print(X_submission.shape)

print(y_submission.shape) # We can see that the submission set contains all the data
batch_size = 32 * strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU

# As a result we will store batches of 256 images into the dataset
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transform matrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([1],dtype='float32')

    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3])

    

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape(tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3])    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape(tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3])

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape(tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3])



    return(rotation_matrix)
def transform(image,label):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated

    DIM = image.shape[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 12. * tf.random.normal([1],dtype='float32')

    shr = 30. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 10. * tf.random.normal([1],dtype='float32') 

    w_shift = 20. * tf.random.normal([1],dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = tf.keras.backend.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = tf.keras.backend.cast(idx2,dtype='int32')

    idx2 = tf.keras.backend.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,1]),label
# Put data in a tensor format for parallelization

def data_augment(image,label):

    #image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

    #image = tf.image.resize_with_crop_or_pad(image, 35, 35) # Add 6 pixels of padding

    #image = tf.image.random_crop(image, size=[28, 28, 1]) # Random crop back to 28x28    

    image = tf.image.random_brightness(image, max_delta=0.3) # Random brightness



    return(image,label)
train_dataset_augment = (

    tf.data.Dataset

    .from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32)))

    .map(data_augment, num_parallel_calls=AUTO)

    .map(transform, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(AUTO)

)



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32)))

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(AUTO)

)



val_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.float32)))

    .batch(batch_size)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(X_test.astype(np.float32))

    .batch(batch_size)

)



submission_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_submission.astype(np.float32), y_submission.astype(np.float32)))

    #.map(data_augment, num_parallel_calls=AUTO)

    .map(transform, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(AUTO)

)
row = 3; col = 4;

all_elements = train_dataset.unbatch()

one_element = tf.data.Dataset.from_tensors(next(iter(all_elements)))

augmented_element = one_element.repeat().batch(row*col)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        image=img[j]

        plt.imshow(image[:,:,0])

    plt.show()

    break
row = 1; col = 1;

all_elements = train_dataset.unbatch()

one_element = tf.data.Dataset.from_tensors(next(iter(all_elements)))

augmented_element = one_element.repeat().map(data_augment).batch(row*col)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        image=img[j]

        plt.imshow(image[:,:,0])

    plt.show()

    break
row = 3; col = 4;

all_elements = train_dataset.unbatch()

one_element = tf.data.Dataset.from_tensors(next(iter(all_elements)))

augmented_element = one_element.repeat().map(transform).batch(row*col)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        image=img[j]

        plt.imshow(image[:,:,0])

    plt.show()

    break
row = 3; col = 4;

all_elements = train_dataset.unbatch()

one_element = tf.data.Dataset.from_tensors(next(iter(all_elements)))

augmented_element = one_element.repeat().map(transform).map(data_augment).batch(row*col)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        image=img[j]

        plt.imshow(image[:,:,0])

    plt.show()

    break
# Parameters

epochs = 100

n_steps = X_train.shape[0]//batch_size
# Define a custom metric

def top_5_categorical_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)
def CNN_model():

    model = Sequential([

        Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28 ,28 ,1)), # Important to specify the shape of the input data in the first layer.

        Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'), # The kernel_size is the grid that will stop at every possible location to extract a patch of surrounding features

        BatchNormalization(),

        Activation('relu'),

        MaxPool2D(pool_size=(2,2)),

        Dropout(0.10), # We are shunting down 25% of the nodes randomly



        Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'), # Same as block 1 but with 64 nodes

        Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),

        BatchNormalization(),

        Activation('relu'),

        MaxPool2D(pool_size=(2,2)),

        Dropout(0.10),

        Flatten(), # Important to start using the 1D fully connected part of the layer



        Dense(256, activation='relu'), # Creating a layer with 256 nodes

        BatchNormalization(),

        Activation('relu'),



        Dense(128, activation='relu'),

        BatchNormalization(),

        Activation('relu'),

        

        

        Dense(64, activation='relu'),

        BatchNormalization(),

        Activation('relu'),

        Dropout(0.20),



        Dense(10, activation='softmax') # We need to end the model with a Dense layer composed of 10 nodes (because 10 numbers from 0 to 9) and with a softmax activation to get a probability

    ])



    return model

# TPU

with strategy.scope():

    model = CNN_model()

    

model.summary()



# Compile the model

model.compile(optimizer = RMSprop(lr=1e-4), 

              loss = "categorical_crossentropy", 

              metrics=["accuracy", top_5_categorical_accuracy])
# Save weights only for best model

checkpointer = ModelCheckpoint(filepath = 'weights_best_MNIST_20.hdf5', 

                               verbose = 2, 

                               save_best_only = True) # This callback will be used to save the model with the best weights



def scheduler(epoch, lr):

    if epoch < 30: # For the first 30 epochs, the learning rate is not changed

        return(lr)

    else: # After 30 epochs the lrdecreases exponentially

        return(lr*math.exp(-0.1))



LRScheduler = LearningRateScheduler(scheduler)



earlystopper = EarlyStopping(monitor='val_loss', min_delta =0, patience=20, verbose=2, mode='min',restore_best_weights=True) # This callback is used to stop the training session if the model does'nt learn anymore
history = model.fit(train_dataset, 

                    steps_per_epoch = n_steps, 

                    epochs = 60, 

                    validation_data=(val_dataset),

                    callbacks = [checkpointer, LRScheduler, earlystopper])
def plot_history(model_history):



    plt.figure(figsize = (20,15))

    

    plt.subplot(221)

    # summarize history for accuracy

    plt.plot(model_history.history['top_5_categorical_accuracy'][5:])

    plt.plot(model_history.history['val_top_5_categorical_accuracy'][5:])

    plt.title('top_3_categorical_accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.grid()

    

    plt.subplot(222)

    # summarize history for accuracy

    plt.plot(model_history.history['accuracy'][5:])

    plt.plot(model_history.history['val_accuracy'][5:])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.grid()

    

    plt.subplot(223)

    # summarize history for loss

    plt.plot(model_history.history['loss'][5:])

    plt.plot(model_history.history['val_loss'][5:])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.grid()

    

    plt.subplot(224)

    # summarize history for lr

    plt.plot(model_history.history['lr'][5:])

    plt.title('learning rate')

    plt.ylabel('lr')

    plt.xlabel('epoch')

    plt.grid()

    

    plt.show()
plot_history(history)
# TPU

with strategy.scope():

    # loading the model with the best validation accuracy

    model.load_weights('weights_best_MNIST_20.hdf5')

    

model.evaluate(val_dataset)
def plot_confusion_matrix(confusion_matrix, 

                          cmap=plt.cm.Reds):

    

    classes = range(10)

    

    plt.figure(figsize=(8,8))

    plt.imshow(confusion_matrix, 

               interpolation='nearest', 

               cmap=cmap)

    plt.title('Confusion matrix')

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = confusion_matrix.max() / 2.

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):

        plt.text(j, i, confusion_matrix[i, j],

                 horizontalalignment="center",

                 color="white" if confusion_matrix[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Predict the values from the validation dataset

y_pred = model.predict(X_val.astype(np.float32))

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred, axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val, axis = 1) 

# compute the confusion matrix

cm = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(cm)
# We recreate a model from scartch and we use the full data available in X_submission and y_submission

earlystopper = EarlyStopping(monitor='loss', min_delta =0, patience=10, verbose=2, mode='min',restore_best_weights=True)



def scheduler(epoch, lr):

    if epoch < 30: # For the first 20 epochs, the learning rate is not changed

        return(lr)

    else: # After ten is decreases exponentially

        return(lr*math.exp(-0.1))



LRScheduler = LearningRateScheduler(scheduler)



# TPU

with strategy.scope():

    # loading the model with the best validation accuracy

    model = CNN_model()



# TPU

with strategy.scope():

    model = CNN_model()



# Compile the model

model.compile(optimizer = RMSprop(lr=1e-4), 

              loss = "categorical_crossentropy", 

              metrics=["accuracy"])



history = model.fit(submission_dataset, 

                    steps_per_epoch = n_steps, 

                    epochs = 60, 

                    callbacks = [LRScheduler, earlystopper])


plt.plot(history.history['accuracy'][10:])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.grid()
# predict results

y_test_pred = model.predict(test_dataset)



# Associate max probability obs with label class

y_test_pred = np.argmax(y_test_pred, axis = 1)

y_test_pred = pd.Series(y_test_pred, name="Label")



submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), y_test_pred], axis = 1)



submission.to_csv("CNN_model_TPU_submission.csv", index = False)