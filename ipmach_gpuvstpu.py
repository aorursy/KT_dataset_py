from tensorflow.python.keras.datasets import mnist

%matplotlib inline

from matplotlib import pyplot as plt

import numpy as np

from tensorflow.python.keras.models import Sequential

import numpy as np

from tensorflow.python.keras.utils import to_categorical 

from tensorflow.python.keras.layers import Input, Dense

from tensorflow.python.keras import layers

from tensorflow.python import keras

import time

import tensorflow as tf
#Use tensorflow/keras  interface generator.

#Generators are use to help train models in different threads and to optimize the RAM we use in the process.

class Data_generator(keras.utils.Sequence):

    #To initialice the object

    #x_set: index of all the files x in the BD [i for i in range(len(BD))]

    #y_set: the same of x_set

    #batch_size: batch_size use for the training process.

    def __init__(self, x_set, y_set, batch_size,data_x,data_y):

        self.x, self.y = x_set, y_set

        self.batch_size = batch_size

        self.data_x = data_x

        self.data_y = data_y

        self.shape = np.array([batch_size])

      

    #Give the number of batch we can take from the DB

    def __len__(self):

        return int(self.x.shape[0] / self.batch_size)

        

    #get a batch of x,y

    #idx: number of the batch

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) *

        self.batch_size]

        batch_y = self.y[idx * self.batch_size:(idx + 1) *

        self.batch_size]

        x = np.take(self.data_x,batch_x, axis =0)

        y = np.take(self.data_y, batch_y, axis = 0)

        return x,y



#Small function to plot the results we get after training a model.

def plot_training_score(history):

  print('Availible variables to plot: {}'.format(history.history.keys()))

  print(" Loss vs epochs")

  plt.figure(1)

  plt.plot( np.arange(len(history.history['loss'])),history.history['loss'])

  print(" Accuracy vs epochs")

  plt.figure(2)

  plt.plot( np.arange(len(history.history['acc'])), history.history['acc'])





# The data, split between train and test sets

# x_train is a list of training images, y_train is a list og training lables

# x_test is a list of test images, y_test is a list of test lables

(x_train_entire, y_train_entire), (x_test, y_test) = mnist.load_data()



# Split x_train_entire and y_train_entire into training and validation

x_train    = x_train_entire[ :50000]

x_validate = x_train_entire[50001: ]

y_train    = y_train_entire[ :50000]

y_validate = y_train_entire[50001: ]



#Making the reshape of the x

(a,b,c)= x_train.shape

x_train_conv = np.reshape(x_train, (a,b,c,1))

(a,_,_) = x_validate.shape

x_validate_conv = np.reshape(x_validate, (a,b,c,1))

(a,_,_) = x_test.shape

x_test_conv = np.reshape(x_test, (a,b,c,1))



#Making y into categorical

y_train_one_hot = to_categorical(y_train)

y_test_one_hot = to_categorical(y_test)

y_validate_one_hot= to_categorical(y_validate)



#Creating index for generators

set_tr_x = np.array([i for i in range(x_train_conv.shape[0])])

set_tr_y = set_tr_x.copy()

set_v_x = np.array([i for i in range(x_validate_conv.shape[0])])

set_v_y = set_v_x.copy()

set_t_x = np.array([i for i in range(x_test_conv.shape[0])])

set_t_y = set_t_x.copy()



#Initializing generators

g_tr = Data_generator(set_tr_x, set_tr_y, 128, x_train_conv, y_train_one_hot)

g_v = Data_generator(set_v_x, set_v_y, 128, x_validate_conv, y_validate_one_hot)

g_t = Data_generator(set_t_x, set_t_y, 128, x_test_conv, y_test_one_hot)
#CNN model

def conv_model(img_width, img_height):

  model = Sequential()  # Initalize a new model

  model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (28,28,1)))

  model.add(layers.MaxPooling2D((2,2)))

  model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

  model.add(layers.Flatten())

  model.add(layers.Dense(64, activation= 'relu'))

  model.add(layers.Dense(10,activation = 'softmax'))

  model.compile(loss='categorical_crossentropy',

              optimizer= tf.train.AdamOptimizer(1e-4) ,

              metrics=['accuracy']) 



  return model





# TODO: Create the model

(width, height) = x_train[0].shape

model = conv_model(width, height)
#Training model with GPU

start = time.time()

history = model.fit_generator(g_tr, epochs =5,steps_per_epoch= 1000,validation_data = g_v, use_multiprocessing= True)

end = time.time()

print("Time: ", end - start)

plot_training_score(history)
#Testing model with GPU

start = time.time()

score, acc = model.evaluate_generator(g_t, use_multiprocessing=True)

end = time.time()

print("Time: ", end - start)

print(score, " ", acc)


# The data, split between train and test sets

# x_train is a list of training images, y_train is a list og training lables

# x_test is a list of test images, y_test is a list of test lables

(x_train_entire, y_train_entire), (x_test, y_test) = mnist.load_data()



# Split x_train_entire and y_train_entire into training and validation

x_train    = x_train_entire[ :50000]

x_validate = x_train_entire[50001: ]

y_train    = y_train_entire[ :50000]

y_validate = y_train_entire[50001: ]



#Making the reshape of the x

(a,b,c)= x_train.shape

x_train_conv = np.reshape(x_train, (a,b,c,1))

(a,_,_) = x_validate.shape

x_validate_conv = np.reshape(x_validate, (a,b,c,1))

(a,_,_) = x_test.shape

x_test_conv = np.reshape(x_test, (a,b,c,1))



#Making y into categorical

y_train_one_hot = to_categorical(y_train)

y_test_one_hot = to_categorical(y_test)

y_validate_one_hot= to_categorical(y_validate)



#Creating index for generators

set_tr_x = np.array([i for i in range(x_train_conv.shape[0])])

set_tr_y = set_tr_x.copy()

set_v_x = np.array([i for i in range(x_validate_conv.shape[0])])

set_v_y = set_v_x.copy()

set_t_x = np.array([i for i in range(x_test_conv.shape[0])])

set_t_y = set_t_x.copy()



#Initializing generators

#Important to multiple * 8 the batch size so we use TPU hardware correctly. 

g_tr = Data_generator(set_tr_x, set_tr_y, 128 * 8, x_train_conv, y_train_one_hot)

g_v = Data_generator(set_v_x, set_v_y, 128 * 8, x_validate_conv, y_validate_one_hot)

g_t = Data_generator(set_t_x, set_t_y, 128 * 8, x_test_conv, y_test_one_hot)
import os

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Input

from tensorflow.python import keras 



#CNN model

def conv_model2():

  visible = keras.layers.Input(shape=(28,28,1))

  conv1 = keras.layers.Conv2D(32,(3,3), activation = 'relu')(visible)

  maxPool1 = keras.layers.MaxPooling2D((2,2))(conv1)

  conv2 = keras.layers.Conv2D(64,(3,3), activation = 'relu')(maxPool1)

  flatten = keras.layers.Flatten()(conv2)

  dense1 = keras.layers.Dense(64, activation= 'relu')(flatten)

  output = keras.layers.Dense(10, activation='softmax')(dense1)

  model = keras.models.Model(inputs=visible, outputs=output)

  model.compile(loss='categorical_crossentropy',

              optimizer= tf.train.AdamOptimizer(1e-4) ,

              metrics=['accuracy'])

  return model

  

  

# TODO: Create the model

(width, height) = x_train[0].shape

#model = conv_model(width, height)

model = conv_model2()





# This address identifies the TPU we'll use when configuring TensorFlow.

TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

tf.logging.set_verbosity(tf.logging.INFO)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(

    model,

    strategy=tf.contrib.tpu.TPUDistributionStrategy(

        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
#Training with TPU

start = time.time()

history = tpu_model.fit_generator(g_tr, epochs =5,steps_per_epoch= 1000,validation_data = g_v) 

end = time.time()

print("Time: ", end - start)

plot_training_score(history)
#Testing with TPU

start = time.time()

score, acc = tpu_model.evaluate_generator(g_t)

end = time.time()

print("Time: ", end - start)

print(score, " ", acc)