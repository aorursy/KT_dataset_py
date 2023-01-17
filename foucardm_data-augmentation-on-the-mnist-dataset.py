import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns # Another data visualisation (prettier than matplotlib)



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
# If you are not using a kaggle notebook with the MNIST attached to it:



# from keras.datasets import mnist

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



print('train shape is: ' + str(df_train.shape))

print('test shape is: ' + str(df_test.shape))
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
X_train = X_train.values.reshape(-1,28,28,1) # (nbr of samples, height, width, channel) Since these are not colored images, there's only one channel

df_test = df_test.values.reshape(-1,28,28,1)
y_label = y_train # Just keeping a copy of the original labels before the one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42) # Here we are creating the training set and the validation set
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
datagen = ImageDataGenerator(rotation_range=10, # Rotating randomly the images up to 25°

                             width_shift_range=0.05, # Moving the images from left to right

                             height_shift_range=0.05, # Then from top to bottom

                             shear_range=0.10, 

                             zoom_range=0.05, # Zooming randomly up to 20%

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
earlystopper = EarlyStopping(monitor='val_loss', min_delta =0, patience=6, verbose=1, mode='min',restore_best_weights=True) # If after 5 epochs (*patience=5*), the validation loss haven't decreased at all (*min_delta=0), the training stage is stopped
model = Sequential([

    Conv2D(filters = 32, kernel_size = (5,5), activation ='relu', input_shape = (28 ,28 ,1)), # Important to specify the shape of the input data in the first layer.

    Conv2D(filters = 32, kernel_size = (5,5), activation ='relu'), # The kernel_size is the grid that will stop at every possible location to extract a patch of surrounding features

    BatchNormalization(),

    Activation('relu'),

    MaxPool2D(pool_size=(2,2)),

    Dropout(0.05), # We are shunting down 25% of the nodes randomly

    

    Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'), # Same as block 1 but with 64 nodes

    Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'),

    BatchNormalization(),

    Activation('relu'),

    MaxPool2D(pool_size=(2,2)),

    Dropout(0.05),

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

    Dropout(0.05),

    

    Dense(10, activation='softmax') # We need to end the model with a Dense layer composed of 10 nodes (because 10 numbers from 0 to 9) and with a softmax activation to get a probability

])



model.summary()
model.compile(optimizer =RMSprop(lr=1e-4) , loss = "categorical_crossentropy", metrics=["acc"])
# Hyperparameters:



EPOCHS = 50 # Number of time the model will see the data

BATCH_SIZE = 86 # Number of images per batch

STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE # Since we are batching, we need to specify when the model should consider that one epoch has been processed.

# A proper way to fix this parameter is to have a number of step equal to the number of data

CALLBACKS = [LRScheduler, earlystopper] # List of the previously defined callbacks



# Fitting the model:

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              epochs = EPOCHS, 

                              validation_data = (X_val,y_val),

                              verbose = 2,

                              steps_per_epoch=STEPS_PER_EPOCH,

                              callbacks=CALLBACKS)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



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
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



y_train = df_train['label'] # y_train will contains the labels

X_train = df_train.drop('label', axis = 1) # X_train contains the data to build the images



X_train /= 255.0

df_test /= 255.0



X_train = X_train.values.reshape(-1,28,28,1) # (nbr of samples, height, width, channel) Since these are not colored images, there's only one channel

df_test = df_test.values.reshape(-1,28,28,1)



y_train = to_categorical(y_train, num_classes = 10)



with strategy.scope():



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

        Dropout(0.05),

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

        Dropout(0.05),



        Dense(10, activation='softmax') # We need to end the model with a Dense layer composed of 10 nodes (because 10 numbers from 0 to 9) and with a softmax activation to get a probability

    ])



model.compile(optimizer =RMSprop(lr=1e-4) , loss = "categorical_crossentropy", metrics=["acc"])





# Hyperparameters:



EPOCHS = 50 # Number of time the model will see the data

BATCH_SIZE = 86 # Number of images per batch

STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE # Since we are batching, we need to specify when the model should consider that one epoch has been processed.

# A proper way to fix this parameter is to have a number of step equal to the number of data

CALLBACKS = [LRScheduler, earlystopper] # List of the previously defined callbacks



# Fitting the model:

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              epochs = EPOCHS,

                              verbose = 2,

                              steps_per_epoch=STEPS_PER_EPOCH,

                              callbacks=CALLBACKS)
# predict results

results = model.predict(df_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns # Another data visualisation (prettier than matplotlib)



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



df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



y_train = df_train['label'] # y_train will contains the labels

X_train = df_train.drop('label', axis = 1) # X_train contains the data to build the images



X_train /= 255.0

df_test /= 255.0



X_train = X_train.values.reshape(-1,28,28,1) # (nbr of samples, height, width, channel) Since these are not colored images, there's only one channel

df_test = df_test.values.reshape(-1,28,28,1)



y_train = to_categorical(y_train, num_classes = 10)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42) # Here we are creating the training set and the validation set



datagen = ImageDataGenerator(rotation_range=10, # Rotating randomly the images up to 25°

                             width_shift_range=0.05, # Moving the images from left to right

                             height_shift_range=0.05, # Then from top to bottom

                             shear_range=0.10, 

                             zoom_range=0.05, # Zooming randomly up to 20%

                             zca_whitening=False,

                             horizontal_flip=False, 

                             vertical_flip=False,

                            fill_mode = 'nearest')



datagen.fit(X_train) # Very important to fit the Generator on the data



dropout_list = [0.06,0.10,0.15,0.20,0.25,0.30,0.35]

dict_acc = {}

dict_val_acc = {}

dict_loss = {}

dict_val_loss = {}



def scheduler(epoch, lr):

    if epoch < 15: # For the first 10 epochs, the learning rate is not changed

        return(lr)

    elif 15 < epoch < 30: # After ten is decreases exponentially

        return(lr*math.exp(-0.1))

    else:

        return(lr*math.exp(-0.2)) # And then decreases even more, still exponentially



LRScheduler = LearningRateScheduler(scheduler)



earlystopper = EarlyStopping(monitor='val_loss', min_delta =0, patience=6, verbose=1, mode='min', restore_best_weights=True) # If after 5 epochs (*patience=5*), the validation loss haven't decreased at all (*min_delta=0), the training stage is stopped



def create_model(dp):

    with strategy.scope():

        

        model = Sequential([

            Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28 ,28 ,1)), # Important to specify the shape of the input data in the first layer.

            Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'), # The kernel_size is the grid that will stop at every possible location to extract a patch of surrounding features

            BatchNormalization(),

            Activation('relu'),

            MaxPool2D(pool_size=(2,2)),

            Dropout(dp), # We are shunting down 25% of the nodes randomly



            Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'), # Same as block 1 but with 64 nodes

            Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),

            BatchNormalization(),

            Activation('relu'),

            MaxPool2D(pool_size=(2,2)),

            Dropout(dp),

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

            Dropout(dp),



            Dense(10, activation='softmax') # We need to end the model with a Dense layer composed of 10 nodes (because 10 numbers from 0 to 9) and with a softmax activation to get a probability

        ])



    model.compile(optimizer =RMSprop(lr=1e-4) , loss = "categorical_crossentropy", metrics=["acc"])

    

    return(model)



EPOCHS = 50 # Number of time the model will see the data

BATCH_SIZE = 86 # Number of images per batch

STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE # Since we are batching, we need to specify when the model should consider that one epoch has been processed.

# A proper way to fix this parameter is to have a number of step equal to the number of data

CALLBACKS = [LRScheduler, earlystopper] # List of the previously defined callbacks



for dp in dropout_list:



    # Fitting the model:

    history = create_model(dp).fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              validation_data = (X_val,y_val),

                              epochs = EPOCHS,

                              verbose = 2,

                              steps_per_epoch=STEPS_PER_EPOCH,

                              callbacks=CALLBACKS)



    dict_acc['acc_for_dp_'+ str(dp)] = history.history['acc']

    dict_val_acc['val_acc_for_dp_'+ str(dp)] = history.history['val_acc']

    dict_loss['loss_for_dp_'+ str(dp)] = history.history['loss']

    dict_val_loss['val_loss_for_dp_'+ str(dp)] = history.history['val_loss']
for key_acc, key_val_acc in zip(dict_acc.keys(), dict_val_acc.keys()):

    plt.figure()

    epochs = range(1, len(dict_acc[key_acc])-2)

    plt.plot(epochs, dict_acc[key_acc][3:], 'bo', label='Training acc')

    plt.plot(epochs, dict_val_acc[key_val_acc][3:],'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.show()
plt.figure()

epochs = range(1, 16)

for key_val_acc in dict_val_acc.keys():

    plt.plot(epochs, dict_val_acc[key_val_acc][len(dict_val_acc[key_val_acc])-15:], label=key_val_acc)

plt.title('Training and validation accuracy')

plt.legend()

plt.show()
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



df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



y_train = df_train['label'] # y_train will contains the labels

X_train = df_train.drop('label', axis = 1) # X_train contains the data to build the images



X_train /= 255.0

df_test /= 255.0



X_train = X_train.values.reshape(-1,28,28,1) # (nbr of samples, height, width, channel) Since these are not colored images, there's only one channel

df_test = df_test.values.reshape(-1,28,28,1)



y_train = to_categorical(y_train, num_classes = 10)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42) # Here we are creating the training set and the validation set



rot_list = [0,5,10,15,20,25,30,35,40,45]



def create_datagen(rot):

    datagen = ImageDataGenerator(rotation_range=rot, # Rotating randomly the images up to 25°

                             width_shift_range=0.05, # Moving the images from left to right

                             height_shift_range=0.05, # Then from top to bottom

                             shear_range=0.10, 

                             zoom_range=0.05, # Zooming randomly up to 20%

                             zca_whitening=False,

                             horizontal_flip=False, 

                             vertical_flip=False,

                            fill_mode = 'nearest')

    return(datagen)







dropout_list = 0.15

dict_acc = {}

dict_val_acc = {}

dict_loss = {}

dict_val_loss = {}



def scheduler(epoch, lr):

    if epoch < 15: # For the first 10 epochs, the learning rate is not changed

        return(lr)

    elif 15 < epoch < 30: # After ten is decreases exponentially

        return(lr*math.exp(-0.1))

    else:

        return(lr*math.exp(-0.2)) # And then decreases even more, still exponentially



LRScheduler = LearningRateScheduler(scheduler)



earlystopper = EarlyStopping(monitor='val_loss', min_delta =0, patience=8, verbose=1, mode='min', restore_best_weights=True) # If after 5 epochs (*patience=5*), the validation loss haven't decreased at all (*min_delta=0), the training stage is stopped



def create_model(rot):

    

    datagen=create_datagen(rot)

    datagen.fit(X_train) # Very important to fit the Generator on the data

    

    with strategy.scope():

        

        model = Sequential([

            Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28 ,28 ,1)), # Important to specify the shape of the input data in the first layer.

            Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'), # The kernel_size is the grid that will stop at every possible location to extract a patch of surrounding features

            BatchNormalization(),

            Activation('relu'),

            MaxPool2D(pool_size=(2,2)),

            Dropout(0.15), # We are shunting down 25% of the nodes randomly



            Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'), # Same as block 1 but with 64 nodes

            Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),

            BatchNormalization(),

            Activation('relu'),

            MaxPool2D(pool_size=(2,2)),

            Dropout(0.15),

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

            Dropout(0.15),



            Dense(10, activation='softmax') # We need to end the model with a Dense layer composed of 10 nodes (because 10 numbers from 0 to 9) and with a softmax activation to get a probability

        ])



    model.compile(optimizer =RMSprop(lr=1e-4) , loss = "categorical_crossentropy", metrics=["acc"])

    

    return(model)



EPOCHS = 50 # Number of time the model will see the data

BATCH_SIZE = 86 # Number of images per batch

STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE # Since we are batching, we need to specify when the model should consider that one epoch has been processed.

# A proper way to fix this parameter is to have a number of step equal to the number of data

CALLBACKS = [LRScheduler, earlystopper] # List of the previously defined callbacks



for rot in rot_list:



    # Fitting the model:

    history = create_model(rot).fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              validation_data = (X_val,y_val),

                              epochs = EPOCHS,

                              verbose = 2,

                              steps_per_epoch=STEPS_PER_EPOCH,

                              callbacks=CALLBACKS)



    dict_acc['acc_for_rot_'+ str(rot)] = history.history['acc']

    dict_val_acc['val_acc_for_rot_'+ str(rot)] = history.history['val_acc']

    dict_loss['loss_for_rot_'+ str(rot)] = history.history['loss']

    dict_val_loss['val_loss_for_rot_'+ str(rot)] = history.history['val_loss']
plt.figure()

epochs = range(1, 16)

for key_val_acc in dict_val_acc.keys():

    plt.plot(epochs, dict_val_acc[key_val_acc][len(dict_val_acc[key_val_acc])-15:], label=key_val_acc)

plt.title('Training and validation accuracy')

plt.legend()

plt.show()
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
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



df_train.sample(frac=1)



y_train = df_train.loc[:df_train.shape[0]//10,['label']] # y_train will contains the labels

X_train = df_train.loc[:df_train.shape[0]//10].drop('label', axis = 1) # X_train contains the data to build the images

y_test = df_train.loc[df_train.shape[0]//10:,['label']]

X_test = df_train.loc[df_train.shape[0]//10:].drop('label', axis = 1)
X_train /= 255.0

X_test /= 255.0

df_test /= 255.0



X_train = X_train.values.reshape(-1,28,28,1) # (nbr of samples, height, width, channel) Since these are not colored images, there's only one channel

df_test = df_test.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



y_label = y_train

y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes = 10)
unique, counts = np.unique(y_label, return_counts=True)

D = dict(zip(unique, counts))



plt.bar(range(len(D)), list(D.values()), align='center')

#plt.xticks(range(len(D)), list(D.keys()))

plt.legend()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42) # Here we are creating the training set and the validation set
y_val_label = np.argmax(y_val,axis = 1)

y_train_label = np.argmax(y_train,axis = 1)
unique, counts = np.unique(y_val_label, return_counts=True)

dic_val = dict(zip(unique, counts))



unique, counts = np.unique(y_train_label, return_counts=True)

dic_train = dict(zip(unique, counts))



plt.figure()

plt.bar(np.arange(len(dic_val))-.2, list(dic_val.values()), align='center', width=0.4)

plt.bar(np.arange(len(dic_train))+.2, list(dic_train.values()), align='center', width=0.4)

#plt.xticks(range(len(D)), list(D.keys()))

plt.legend()
rot_list = [0.5,10,20]

w_list = [0,0.05,0.1]

h_list = [0,0.05,0.1]

sh_list = [0,0.1,0.2]

zoom_list = [0,0.05,0.1]



def create_datagen(rot,w,h,sh,zoom):

    datagen = ImageDataGenerator(rotation_range=rot, # Rotating randomly the images up to 25°

                             width_shift_range=w, # Moving the images from left to right

                             height_shift_range=w, # Then from top to bottom

                             shear_range=sh, 

                             zoom_range=zoom, # Zooming randomly up to 20%

                             zca_whitening=False,

                             horizontal_flip=False, 

                             vertical_flip=False,

                            fill_mode = 'nearest')

    return(datagen)



dict_acc = {}

dict_val_acc = {}

dict_loss = {}

dict_val_loss = {}

dict_test_acc = {}

k = 3

num_val_samples = len(X_train) // k



df = pd.DataFrame(columns=['rotation', 'width', 'height', 'shear', 'zoom', 'val_acc', 'test_loss', 'test_acc'])



def scheduler(epoch, lr):

    if epoch < 10: # For the first 10 epochs, the learning rate is not changed

        return(lr)

    elif 10 < epoch < 20: # After ten is decreases exponentially

        return(lr/2)

    else:

        return(lr*math.exp(-0.1)) # And then decreases even more, still exponentially



LRScheduler = LearningRateScheduler(scheduler)



earlystopper = EarlyStopping(monitor='val_loss', min_delta =0, patience=10, verbose=1, mode='min', restore_best_weights=False) # If after 5 epochs (*patience=5*), the validation loss haven't decreased at all (*min_delta=0), the training stage is stopped



def create_model(rot,w,h,sh,zoom):

    

    datagen=create_datagen(rot,w,h,sh,zoom)

    datagen.fit(X_train) # Very important to fit the Generator on the data

    

    with strategy.scope():

        

        model = Sequential([

            Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28 ,28 ,1)), # Important to specify the shape of the input data in the first layer.

            Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'), # The kernel_size is the grid that will stop at every possible location to extract a patch of surrounding features

            BatchNormalization(),

            Activation('relu'),

            MaxPool2D(pool_size=(2,2)),

            Dropout(0.15), # We are shunting down 25% of the nodes randomly



            Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'), # Same as block 1 but with 64 nodes

            Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'),

            BatchNormalization(),

            Activation('relu'),

            MaxPool2D(pool_size=(2,2)),

            Dropout(0.15),

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

            Dropout(0.15),



            Dense(10, activation='softmax') # We need to end the model with a Dense layer composed of 10 nodes (because 10 numbers from 0 to 9) and with a softmax activation to get a probability

        ])



    model.compile(optimizer = RMSprop(lr=1e-3) , loss = "categorical_crossentropy", metrics=["acc"])

    

    return(model)



EPOCHS = 50 # Number of time the model will see the data

BATCH_SIZE = 86 # Number of images per batch

STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE # Since we are batching, we need to specify when the model should consider that one epoch has been processed.

# A proper way to fix this parameter is to have a number of step equal to the number of data

CALLBACKS = [LRScheduler, earlystopper] # List of the previously defined callbacks



for rot in rot_list:

    for w in w_list:

        for sh in sh_list:

            for zoom in zoom_list:

                for h in h_list:

                    all_scores = []

                    for i in range(k):

                        val_data = X_train[i*num_val_samples:(i+1)*num_val_samples]

                        val_targets = y_train[i*num_val_samples:(i+1)*num_val_samples]



                        partial_train_data = np.concatenate(

                        [X_train[:i*num_val_samples],

                        X_train[(i+1)*num_val_samples:]],

                        axis=0)



                        partial_train_targets = np.concatenate(

                        [y_train[:i*num_val_samples],

                        y_train[(i+1)*num_val_samples:]],

                        axis=0)



                        model = create_model(rot,w,h,sh,zoom)

                        model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                                                                  validation_data = (X_val,y_val),

                                                                  epochs = EPOCHS,

                                                                  verbose = 0,

                                                                  steps_per_epoch=STEPS_PER_EPOCH,

                                                                  callbacks=CALLBACKS)

                        val_mse, val_mae = model.evaluate(X_val, y_val, verbose = 0)

                        all_scores.append(val_mae)

                    val_score = np.mean(all_scores)

                    test_loss, test_score = model.evaluate(X_test, y_test)

                    

                    df = df.append({'rotation': rot, 'width':w, 'height':h, 'shear':sh, 'zoom':zoom, 'val_acc':val_score, 'test_loss':test_loss, 'test_acc':test_score}, ignore_index=True)

                    

                    print('acc_'+ str(rot) + ' width_' + str(w)+ ' height_' + str(h)+ ' shear_' + str(sh)+ ' zoom_' + str(zoom) + str(test_score))