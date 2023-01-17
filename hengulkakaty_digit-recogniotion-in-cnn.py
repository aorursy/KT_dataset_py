import numpy as np
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

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd

df1 = pd.read_csv('../input/train.csv')
df2 = pd.read_csv('../input/test.csv')
train_data = np.array(df1)
test_data  = np.array(df2)

X_train_orig    = np.asarray( train_data[:, 1::], dtype=np.float32 )
Y_train_orig    = np.asarray( train_data[:, 0], dtype=np.float32 )

X_test_orig     = np.asarray( test_data[:, 0::], dtype=np.float32 )
X_train = X_train_orig / 255.

X_test = X_test_orig / 255.
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)

Y_train = Y_train_orig.reshape(-1, 1)

from keras.utils import to_categorical

Y_train = to_categorical(Y_train, num_classes=10)
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
def Keras_Model(input_shape):    
    
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((1, 1))(X_input) 
    
    print('Size of X after Zero padding', X.shape)
    
    X = Conv2D(32, (5, 5), strides = (1, 1), padding = 'same', name = 'conv0')(X) 
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(32, (5, 5), strides = (1, 1), padding = 'same', name = 'conv1')(X) 
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
                           
    X = MaxPooling2D((2, 2), name='max_pool_1')(X)
                           
    X = Dropout(0.25)(X)
                           
    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2')(X) 
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv3')(X) 
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
                           
    X = MaxPooling2D((2, 2), name='max_pool_2')(X)                       
                           
    X = Dropout(0.25)(X)                       
 
    X = Flatten()(X)
    X = Dense(1024, activation='relu', name='fc0')(X) 
    X = Dense(256, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)                       
    X = Dense(10, activation='softmax', name='fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='model')
    
    return model
Keras_Model = Keras_Model(X_train.shape[1:4])
from keras.optimizers import Adam
optimizer = Adam(lr=0.0001, epsilon=1e-08, decay=0.0)
Keras_Model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
Keras_Model.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 64)
preds = Keras_Model.evaluate(X_train, Y_train)
print ("Loss = " + str(preds[0]))
print ("Train set Accuracy = " + str(preds[1]))
classes = Keras_Model.predict(X_test, batch_size=64)
np.set_printoptions(suppress=True)
classes
class_test_set = np.argmax(classes, axis = 1)
prediction = pd.DataFrame()
prediction['ImageId'] = np.asarray(range(1,28001))
prediction['Label'] = class_test_set

prediction.to_csv('submission_without_Augmentation.csv', index = False)
