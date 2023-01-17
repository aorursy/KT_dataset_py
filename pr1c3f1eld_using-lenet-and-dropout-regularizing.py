import numpy as np

import pandas as pd

import torch

import torchvision

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

#from sklearn.metrics import confusion_matrix

#import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

#from keras.preprocessing.image import ImageDataGenerator

#from keras.callbacks import ReduceLROnPlateau

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

import tensorflow as tf



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow





sns.set(style='white', context='notebook', palette='deep')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



Y_train_orig = train["label"]

X_train_orig = train.drop(labels = ["label"],axis = 1)



del train



g = sns.countplot(Y_train_orig)



Y_train_orig.value_counts()
#np.size((Y_train_orig.values))

#Y_train_orig.shape

test.shape
X_train_orig = X_train_orig/255

test = test/255
#Reshape

X_train_orig = X_train_orig.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train_orig = to_categorical(Y_train_orig, num_classes = 10)

random_seed = 2

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_orig, Y_train_orig, test_size = 0.1,random_state = random_seed)
def DigitModel(input_shape):

  X_input = Input(input_shape)

  X = X_input

  

  #Layer 1 - Conv>BN>RELU>MaxPool

  

  X = Conv2D(6,(5,5),strides = (1,1), name = 'conv0', padding = 'same')(X)

  X = BatchNormalization(axis = 3, name = 'bn0')(X)

  X = Activation('relu')(X)

  X = Dropout(0.5)(X)

  X = MaxPooling2D((2,2),strides = (2,2), name = 'maxpool0')(X)

  

  

  #Layer 2 - Conv>BN>RELU>MaxPool

  

  X = Conv2D(16,(5,5),strides = (1,1), name = 'conv1', padding = 'valid')(X)

  X = BatchNormalization(axis = 3, name = 'bn1')(X)

  X = Activation('relu')(X)

  X = Dropout(0.5)(X)

  X = MaxPooling2D((2,2),strides = (2,2), name = 'maxpool1')(X)

  

  #FC layers with flattening

  

  X = Flatten()(X)

  X = Dense(120, activation = 'relu', name = 'fc1')(X)

  X = Dropout(0.5)(X)

  X = Dense(84, activation = 'relu', name = 'fc2')(X)

  X = Dropout(0.5)(X)

  X = Dense(10, activation = 'softmax', name = 'preds')(X)

  

  #model instance creator

  model = Model(inputs = X_input, outputs = X, name = 'DigitModel')

  

  return model
#Initialize the model



### START CODE HERE ### (1 line)

input_shape = X_train.shape[1:4]

recog = DigitModel(input_shape)

### END CODE HERE ###
#Compile model to configure learning process

recog.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])
#train the model

recog.fit(x=X_train, y=Y_train, epochs = 30, batch_size = 86)
preds = recog.evaluate(x = X_dev, y = Y_dev)

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
y_test_set = recog.predict(test)



#Getting the required desired matrix from max_indices

results = np.argmax(y_test_set,axis = 1)

#print(results)



#Creating the submission Dataframe

result = pd.DataFrame({'Label' : results })

result_id = pd.DataFrame({'ImageId' : range(1,28001)})

#print(result_id)

result = pd.concat([result_id,result], axis = 1)

#print(result)



#Saving into a CSV file

result.to_csv("submission.csv", index = False)

print(result)