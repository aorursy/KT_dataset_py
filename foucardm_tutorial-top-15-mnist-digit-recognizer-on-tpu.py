import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns # Another data visualisation (prettier than matplotlib)

import itertools



import math # We need this module to use mathematical function such as EXP()



from sklearn.model_selection import train_test_split # Because we will split the training data into a training set and a validation set.

from sklearn.metrics import confusion_matrix # Usefull at the end to see where the model makes mistakes



from keras.utils.np_utils import to_categorical # Because we will convert lebels to one-hot-encoding

from keras.models import Sequential # The type of keras model we'll be using

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation # All the different layers that we need to build the model

from keras.optimizers import RMSprop # Optimizer for the model

from keras.preprocessing.image import ImageDataGenerator # This will help to make data augmentation and batches of images

from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping # To make a model a bit more sophisticated

# Import to loas the previous callbacks from TensorFlow and not from keras !



sns.set(palette='muted') # To build pretty graphs
!ls /kaggle/input/ # The dataset is already attached to this notebook, we can just load the train and test sets form there
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



print('train shape is: ' + str(df_train.shape))

print('test shape is: ' + str(df_test.shape))
df_train.head()
y_train = df_train['label'] # y_train will contains the labels

X_train = df_train.drop('label', axis = 1) # X_train contains the data to build the images
print('shape of the training label dataset: ' + str(y_train.shape))

print('shape of the training image dataset: ' + str(X_train.shape))
X_train.loc[0].value_counts().head() # Looks like pixels values range from 0 to 255. We'll need to normalize these. Moreover, their type is *int64* we need to convert them to float.

# On the fisrt picture we can see that 687 out of the 784 pixels are completely black

# And that around 30 are very bright

# The others should be due to some kind of halo effect where pixels close to the bright ones are a bit ligthen up
X_train /= 255.0

df_test /= 255.0 # Test data should always be normalized the exact same way the training data have been normalized.
X_full = X_train # We'll need this to have a dataset with all the input data

y_full = y_train
y_label = y_train # Just keeping a copy of the original labels before the one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)

y_full = to_categorical(y_full, num_classes = 10)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42) # Here we are creating the training set and the validation set
def reshape_data(data):

    return(data.values.reshape(-1,28,28,1))
X_train = reshape_data(X_train)

X_test = reshape_data(df_test)

X_val = reshape_data(X_val)

X_full = reshape_data(X_full)
def decode(label): # Since we have encoded the labels,we need to decode it to access to the original number.

    maxi = max(label)

    for i in range(len(label)+1):

        if label[i] == maxi:

            return(i)

    

plt.figure()

for i in range(0,9):

    plt.subplot(330 + 1 + i) # Create the 3x3 image grid

    plt.axis('off')

    plt.imshow(X_train[i][:,:,0], cmap='gray')

    plt.title('number = ' + str(decode(y_train[i])))

plt.show()
datagen = ImageDataGenerator(rotation_range=11, # Rotating randomly the images up to 25°

                             width_shift_range=0.08, # Moving the images from left to right

                             height_shift_range=0.08, # Then from top to bottom

                             shear_range=0.10, 

                             zoom_range=0.07, # Zooming randomly up to 20%

                             zca_whitening=False,

                             horizontal_flip=False, 

                             vertical_flip=False,

                            fill_mode = 'nearest')



datagen.fit(X_train) # Very important to fit the Generator on the data



for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):

    for i in range(0,9):

        plt.subplot(330 + 1 + i)

        plt.axis('off')

        plt.imshow(X_batch[i][:,:,0], cmap='gray')

        plt.title('number = ' + str(decode(y_batch[i])))

    break

# Since we are now batching, we won't get the exact same images from the last exemple.
def scheduler(epoch, lr):

    if epoch < 15: # For the first 10 epochs, the learning rate is not changed

        return(lr)

    elif 15 < epoch < 30: # After ten is decreases exponentially

        return(lr*math.exp(-0.1))

    else:

        return(lr*math.exp(-0.2)) # And then decreases even more, still exponentially



LRScheduler = LearningRateScheduler(scheduler)
earlystopper = EarlyStopping(monitor='loss', min_delta =0, patience=6, verbose=1, mode='min',restore_best_weights=True) # If after 8 epochs (*patience=8*), the validation loss haven't decreased at all (*min_delta=0), the training stage is stopped
import tensorflow as tf



# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

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
def create_model():

    with strategy.scope(): # This line of code is very important if you want to run your full code on TPU or GPU. Make sur the model is defined with the "with" loop.

        

        model = Sequential([

                    Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28 ,28 ,1)), # Important to specify the shape of the input data in the first layer.

                    Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'), # The kernel_size is the grid that will stop at every possible location to extract a patch of surrounding features

                    BatchNormalization(),

                    Activation('relu'),

                    MaxPool2D(pool_size=(2,2)),

                    Dropout(0.05), # We are shunting down 25% of the nodes randomly



                    Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'), # Same as block 1 but with 64 nodes

                    Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),

                    BatchNormalization(),

                    Activation('relu'),

                    MaxPool2D(pool_size=(2,2)),

                    Dropout(0.12),

                    Flatten(), # Important to start using the 1D fully connected part of the layer



                    Dense(256, activation='relu'), # Creating a layer with 256 nodes

                    BatchNormalization(),

                    Activation('relu'),



                    Dense(128, activation='relu'),

                    BatchNormalization(),

                    Activation('relu'),



                    Dense(84, activation='relu'),

                    BatchNormalization(),

                    Activation('relu'),

                    Dropout(0.06),



                    Dense(10, activation='softmax') # We need to end the model with a Dense layer composed of 10 nodes (because 10 numbers from 0 to 9) and with a softmax activation to get a probability

        ])



    model.compile(optimizer =RMSprop(lr=1e-4) , loss = "categorical_crossentropy", metrics=["acc"])

    

    return(model)



create_model().summary()
EPOCHS = 80 # Number of time the model will see the data

BATCH_SIZE = 86 # Number of images per batch

STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE # Since we are batching, we need to specify when the model should consider that one epoch has been processed.

# A proper way to fix this parameter is to have a number of step equal to the number of data

CALLBACKS = [LRScheduler, earlystopper] # List of the previously defined callbacks



model = create_model() # Here we build the model by calling the previous function.



history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              validation_data = (X_val,y_val),

                              epochs = EPOCHS,

                              verbose = 2,

                              steps_per_epoch=STEPS_PER_EPOCH,

                              callbacks=CALLBACKS)
acc = history.history['acc'][3:] # We won't display the 3 first epochs in order to have a more precise view of the last points.

val_acc = history.history['val_acc'][3:]

loss = history.history['loss'][3:]

val_loss = history.history['val_loss'][3:]



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show() 
# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)



sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='BuPu')

plt.title('Confusion matrix MNIST')
datagen.fit(X_full) # Very important to fit the Generator on the data



model_submission = create_model()



model_submission.fit_generator(datagen.flow(X_full,y_full, batch_size=BATCH_SIZE), # No more validation set for the accuracy here !

                              epochs = EPOCHS,

                              verbose = 2,

                              steps_per_epoch=STEPS_PER_EPOCH,

                              callbacks=CALLBACKS)
# predict results

results = model_submission.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)