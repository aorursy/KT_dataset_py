# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2 

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
classes = len(emotions)
print(classes)
import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
K.set_image_data_format("channels_last")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import math
import h5py
import tensorflow as tf
from tensorflow.keras import callbacks


%matplotlib inline
import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

import pandas as pd
import numpy as np
import cv2

with strategy.scope():
#     data = pd.read_csv(r'/kaggle/input/data1.csv',names = ['image','image2'])
    label = pd.read_csv(r'/kaggle/input/coolland/labeldata1.csv',names = ['label'])
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]
train_set_x_orig=[]
for id in range(24942):
    image = cv2.imread('/kaggle/input/coolland/landmarks2/landmarks2/'+str(id)+'.png')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(48,48))
    array = np.array(image)
    train_set_x_orig.append(array)
    
# available_transformations = {
#     'rotate': random_rotation,
#     'horizontal_flip': horizontal_flip
# }

# for id in range(24942): 
#     image = cv2.imread('/kaggle/input/landmarks2/landmarks2/'+str(id)+'.png') 
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image,(48,48))
#     array = np.array(image)
#     key = random.choice(list(available_transformations))
#     array2 = available_transformations[key](array)
#     print(array2.shape)
#     train_set_x_orig.append(array2)
available_transformations = {
    'rotate': random_rotation,
    'horizontal_flip': horizontal_flip
}

for id in range(24942): 
    image = cv2.imread('/kaggle/input/coolland/landmarks2/landmarks2/'+str(id)+'.png') 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(48,48))
    array = np.array(image)
    key = random.choice(list(available_transformations))
    array2 = available_transformations[key](array)
    train_set_x_orig.append(array2)
train_set_y_orig=[]
a = []
train_set_y_orig = np.array(label.label).astype(int)
# a = np.array(label.label).astype(int)
# train_set_y_orig = np.append(train_set_y_orig,a)

print(train_set_y_orig.shape[0])
print(train_set_y_orig)
# print(train_set_x_orig)

from keras.utils import np_utils
# print(train_set_x_orig)
train_set_x_orig = np.array(train_set_x_orig)/255 #putting data into a numpy array and normalizing the data by dividing with 255
train_set_x_orig = np.reshape(train_set_x_orig,(24942,48,48,1)) # '1' is for grayscale and for color it would be '3' & data.shape returns the number of photos to create that big matrix
train_set_y_orig = np.array(train_set_y_orig)
train_set_y_orig = np_utils.to_categorical(train_set_y_orig, num_classes = classes)
# print(Y_train.shape)
# print(Y_test.shape)
from keras import regularizers
from keras.regularizers import l2

def Moodel(input_shape):
   
    
    X_input = Input(input_shape)
    
    #Block 1 -- conv2d - MaxPooling
    
    X = ZeroPadding2D((3,3))(X_input)
    
    X = Conv2D(64,(3,3), strides = (1,1), name = 'conv0')(X)
    X = BatchNormalization(axis=3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    #MaxPool
    
    X = MaxPooling2D((3,3), strides = (1,1), name = 'max_pool_0')(X)
    
    
    #Block2 -- conv2D -- maxpool2D
    
    X = Conv2D(64,(3,3), name = 'conv1')(X)
    X = BatchNormalization(axis=3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    #MaxPool
    
    X= MaxPooling2D((2,2), strides = (2,2), name = 'max_pool_1')(X)
    
    
    #Block3 -- conv2D -- MaxPool2D
    
    X = Conv2D(128,(3,3), name ="conv2")(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation("relu")(X)
    
    #MaxPool
    
#     X = MaxPooling2D((3,3), strides = (1,1), name = 'max_pool_2')(X)
    
    
    #Block4 -- conv2D -- MaxPool2D
    
    X = Conv2D(256,(3,3), name ="conv3")(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation("relu")(X)
    
    #MaxPool
    
    X = MaxPooling2D((2,2), strides = (2,2), name = 'max_pool_3')(X)
    
    
    #Block5 -- conv2D -- MaxPool2D
    
#     X = Conv2D(512,(3,3), name ="conv4")(X)
#     X = BatchNormalization(axis = 3, name = 'bn4')(X)
#     X = Activation("relu")(X)
    
#     #MaxPool
    
    X = MaxPooling2D((2,2), strides = (2,2), name = 'max_pool_4')(X)
    
    #Block6 --flatten --dense
    
    X = layers.Dropout(0.6)(X)
    X = Flatten()(X)
#     X = Dense(128, activation = 'relu', name = 'fc',kernel_regularizer=l2(0.02), bias_regularizer=l2(0.03))(X)
    X = Dense(32, activation = 'relu', name = 'fc1',kernel_regularizer=l2(0.02), bias_regularizer=l2(0.03))(X)
    X = Dense(32, activation = 'relu', name = 'fc2',kernel_regularizer=l2(0.02), bias_regularizer=l2(0.03))(X)
    X = Dense(7, activation = 'softmax', name = 'fc3',kernel_regularizer=l2(0.02), bias_regularizer=l2(0.03))(X)
    
   
    
     #Block5 
    
    model = Model(inputs = X_input, output = X, name = 'Moodel')
    
    return model

moodel = Moodel(train_set_x_orig.shape[1:])

moodel.summary()
my_callback = [tf.keras.callbacks.EarlyStopping(monitor = 'acc', patience = .90)]
moodel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
with strategy.scope():
    history = moodel.fit(train_set_x_orig,train_set_y_orig, epochs = 200, batch_size=200, verbose = 1, validation_split = 0.2, callbacks = my_callback)
moodel.save('HappyModel2')
# moodel.evaluate(X_test,Y_test)
import pandas as pd
array = pd.read_csv(r'/kaggle/input/labeldata.csv')
print(array)
