# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

import keras.backend as K





from keras.models import Model, Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from keras.optimizers import Adam, SGD

from keras.callbacks import Callback, TerminateOnNaN, ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard, ReduceLROnPlateau 
train = pd.read_csv('../input/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashion-mnist_test.csv')



train.head(10)
req_df = train.copy(deep = True)

req_df.drop(['label'], axis = 1, inplace = True)

req_df.head(5)



req_df2 = test.copy(deep=True)

req_df2.drop(['label'], axis=1, inplace=True)
# converting dataframe to np ndarray



trainimages = req_df.values

print(trainimages.shape)



testimages = req_df2.values

print(testimages.shape)
trainimages = np.reshape(trainimages, newshape = (60000, 28, 28))

testimages = np.reshape(testimages, newshape=(10000, 28, 28))
import matplotlib.pyplot as plt

plt.imshow(trainimages[2])
# normalizing the data



xtrain = train.copy(deep = True)



xtest = test.copy(deep = True)



columns = []



for item in xtrain:

    if item != 'label':

        columns += [item]
for item in columns:

    xtrain[item] = xtrain[item]/255.0

    xtest[item] = xtest[item]/255.0
# preparing the target data



target_cols = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']



ytrain = pd.DataFrame(columns = target_cols)

ytest = pd.DataFrame(columns = target_cols)



ytrain['label'] = xtrain['label']

ytest['label'] = xtest['label']



counter = 0



for item in target_cols:

    ytrain[item] = ytrain['label'].apply(lambda x : x == counter)

    ytest[item] = ytest['label'].apply(lambda x : x == counter)

    counter += 1



ytrain.drop(['label'], axis = 1, inplace = True)

xtrain.drop(['label'], axis = 1, inplace = True)



ytest.drop(['label'], axis = 1, inplace = True)

xtest.drop(['label'], axis = 1, inplace = True)



ytrain.head()
ytrain = ytrain.values.astype(int)

ytest = ytest.values.astype(int)



print(ytrain.shape)
# callbacks are used to stop training when a desired objective is fulfilled



class myCallback(Callback):

    def on_epoch_end(self, epoch, logs = {}):

        if logs.get('val_acc') > 0.85:

            print('Reached 80% accuracy so cancelling training')

            self.model.stop_training = True

            



# terminates training when a NaN loss is encountered

terminateOnNaN = TerminateOnNaN()

            

# the model checkpoints will be saved with the epoch number and the validation loss in the filename

modelcheckpoint = ModelCheckpoint(filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor = 'val_acc', 

                                 verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', 

                                  period = 1)



# Stop training when a monitored quantity has stopped improving

earlystopping = EarlyStopping(monitor = 'val_acc', min_delta = 0, patience = 0, verbose = 0, mode = 'auto', 

                             baseline = 0.60, restore_best_weights = False)



# LearningRateScheduler



# TensorBoard basic visualizations

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=1024, write_graph=True, 

                          write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, 

                          embeddings_metadata=None, embeddings_data=None, update_freq='epoch')





# Reduce learning rate when a metric has stopped improving



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', 

                              min_delta=0.0001, cooldown=0, min_lr=0)



# CSVLogger : Callback that streams epoch results to a csv file



# LambdaCallback : Callback for creating simple, custom callbacks on-the-fly

# baseline model



model = Sequential([Dense(784, input_shape = (784, ), activation = 'relu'),

                   Dense(128, activation = 'relu'),

                   Dense(10, activation = 'softmax')])



callbacks = myCallback()

model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(xtrain, ytrain, epochs = 10, batch_size = 1024, validation_data = (xtest, ytest), 

          callbacks = [callbacks, terminateOnNaN,  modelcheckpoint, earlystopping, tensorboard, reduce_lr])
# alternative way of training with sparse_categorical_crossentropy loss



ytrain2 = train['label'].values.astype(int) # Here ytrain2 is a 1 dimensional array

ytest2 = test['label'].values.astype(int)



model2 = Sequential([Dense(784, input_shape = (784, ), activation = 'relu'),

                   Dense(128, activation = 'relu'),

                   Dense(10, activation = 'softmax')])



# the number of softmax outputs must be equal to the number of classes



model2.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model2.fit(xtrain, ytrain2, epochs = 10, batch_size = 1024)
# how does a softmax unit work



# the output layer can be modelled as Dense(10, activation = 'softmax')



# the output is a vector of 10 real-valued outputs



# y-train must also be a vector of 10 binary values



# each element of the output vector is a probability that the input example belongs to the corresponding 

# output class



# the sum of these probabilities should add up to 1



# the loss function generally used with softmax units is categorical_crossentropy
model.evaluate(xtest, ytest)
y_preds = model.predict(xtest)

print(y_preds.shape)
print(y_preds[0])

print(y_preds[0].sum())

print(ytest[0])
# use of generator



'''

Trains the model on data generated batch-by-batch by a Python generator (or an instance of Sequence).



The generator is run in parallel to the model, for efficiency. 

For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training 

your model on GPU.



'''
# Convolutional Neural Network Model



# the filter/kernel weights are learnt in a conv-net



trainimages = np.reshape(trainimages, newshape=(60000,28,28,1))

testimages = np.reshape(testimages, newshape=(10000,28,28,1))



conv_model = Sequential([

    Conv2D(filters=64, kernel_size=(3,3), input_shape=(28,28,1),strides=(1,1), padding='valid', data_format=None, 

           activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 

           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 

           bias_constraint=None),

    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'), 

    # the third dimension of the kernel is same as that of the input, it is automatically inferred

    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')

])



optimizer = Adam(lr = 0.0001)



conv_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

conv_model.fit(trainimages, ytrain2, epochs=10, batch_size=1024, validation_data=(testimages, ytest2), 

               callbacks=[terminateOnNaN])
conv_model.summary()



# In the first Conv2D layer, number of parameters = 640

# 64 3x3 filters with 3x3=9 trainable weights and 1 bias term each

# Total number of parameters = (9+1)*64 = 640
# checking if reshaping does not transform the data



checkim = np.reshape(trainimages, newshape=(60000,28,28))

plt.imshow(checkim[5])
# visualizing the convolution and the pooling layers' output



print(ytrain2[:100])
conv_model.layers
conv_model.layers[1].output



# ? indicates that this entry will be equal to the batch size
conv_model.input
import matplotlib.pyplot as plt



f, axarr = plt.subplots(3, 4)

im1 = 5

im2 = 6

im3 = 8

conv_num = 0



layer_outputs = [layer.output for layer in conv_model.layers]

activation_model = Model(inputs = conv_model.input, outputs = layer_outputs)



for x in range(0, 4):

    f1 = activation_model.predict(trainimages[im1].reshape(1,28,28,1))[x]

    axarr[0,x].imshow(f1[0,:,:,conv_num], cmap = 'inferno')

    axarr[0,x].grid(False)

    f2 = activation_model.predict(trainimages[im2].reshape(1,28,28,1))[x]

    axarr[1,x].imshow(f2[0,:,:,conv_num], cmap = 'inferno')

    axarr[1,x].grid(False)

    f3 = activation_model.predict(trainimages[im3].reshape(1,28,28,1))[x]

    axarr[2,x].imshow(f3[0,:,:,conv_num], cmap = 'inferno')

    axarr[2,x].grid(False)

    
activation_model.summary()
activation_model.output