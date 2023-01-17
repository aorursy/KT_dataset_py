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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2 

emotions = ['Angry','Disgust' ,'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
classes = len(emotions)
print(classes)
import pandas as pd
import numpy as np
import cv2

#with tpu_strategy.scope():
#     data = pd.read_csv(r'/kaggle/input/data1.csv',names = ['image','image2'])
def1 = pd.read_csv(r'../input/fe2013landmarks/labeldata1.csv',names = ['label'])
def1["label"] = def1.iloc[:,0:]
def1
def1.replace(to_replace ={'label': {1:0}}, inplace = True)
def1.replace(to_replace ={'label': {2:0}}, inplace = True)
def1.replace(to_replace ={'label': {3:1}}, inplace = True)
def1.replace(to_replace ={'label': {4:2}}, inplace = True)
def1.replace(to_replace ={'label': {5:3}}, inplace = True)
def1.replace(to_replace ={'label': {6:4}}, inplace = True)
train_set_x_orig=[]
#with tpu_strategy.scope():
for id in range(24942): 
    image = cv2.imread('../input/ldzipp/landmarks2/'+str(id)+'.png') 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(48,48))
    array = np.array(image)
    print(array.shape)
    train_set_x_orig.append(array)
train_set_x_orig = np.array(train_set_x_orig)
print(train_set_x_orig.shape)
train_set_x_orig = train_set_x_orig.reshape((24942,48,48,1))
print(train_set_x_orig.shape)
train_set_y_orig = np.array(def1.label).astype(int)
print(train_set_y_orig.shape[0])
print(train_set_y_orig)
from sklearn.model_selection import train_test_split
train_set_x_orig, test_set_x_orig, train_set_y_orig, test_set_y_orig = train_test_split(train_set_x_orig, train_set_y_orig,shuffle = True, stratify = train_set_y_orig, random_state = 2020, test_size = 0.2 )
print(train_set_x_orig.shape)
print(test_set_x_orig.shape)
print(test_set_x_orig.shape)
print(test_set_y_orig.shape)
# Normalize image vectors
X_train = train_set_x_orig/255
X_test = test_set_x_orig/255

# Reshape
Y_train = train_set_y_orig.T
Y_test = test_set_y_orig.T


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
from keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = classes)
Y_test = to_categorical(Y_test, num_classes = classes)
print(Y_train.shape)
print(Y_test.shape)
from keras import regularizers
#with tpu_strategy.scope():
def Moodel1(input_shape):
   
    
    X_input = Input(input_shape)
    
    #Block 1 -- conv2d - MaxPooling
    
    X = ZeroPadding2D((3,3))(X_input)
    
    X = Conv2D(32,(3,3), strides = (1,1), name = 'conv0')(X)
    X = BatchNormalization(axis=3, name = 'bn0')(X)
    X = Activation('relu')(X)
    #X = layers.Dropout(0.2)(X)
    #MaxPool
    
    X = MaxPooling2D((3,3), strides = (1,1), name = 'max_pool_0')(X)
    
    
    #Block2 -- conv2D -- maxpool2D
    
    X = Conv2D(64,(3,3), name = 'conv1')(X)
    X = BatchNormalization(axis=3, name = 'bn1')(X)
    X = Activation('relu')(X)
    #X = layers.Dropout(0.2)(X)
    #MaxPool
    
    X= MaxPooling2D((2,2), strides = (2,2), name = 'max_pool_1')(X)
    
    
    #Block3 -- conv2D -- MaxPool2D
    
    X = Conv2D(80,(3,3), name ="conv2")(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation("relu")(X)
    #X = layers.Dropout(0.2)(X)
    #MaxPool
    
    X = MaxPooling2D((3,3), strides = (1,1), name = 'max_pool_2')(X)
    
    
    #Block4 -- conv2D -- MaxPool2D
    
    X = Conv2D(160,(3,3), name ="conv3")(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation("relu")(X)
    #X = layers.Dropout(0.2)(X)
    #MaxPool
    
    X = MaxPooling2D((2,2), strides = (2,2), name = 'max_pool_3')(X)
    
    
    #Block5 -- conv2D -- MaxPool2D
    
    X = Conv2D(320,(2,2), name ="conv4")(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation("relu")(X)
    #X = layers.Dropout(0.2)(X)
    #MaxPool
    
    X = MaxPooling2D((2,2), strides = (2,2), name = 'max_pool_4')(X)
    
    #Block6 --flatten --dense
    
    X = Conv2D(512,(1,1), name ="conv5")(X)
    X = BatchNormalization(axis = 3, name = 'bn5')(X)
    X = Activation("relu")(X)
    
    X = layers.Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(100, activation = 'relu', name = 'fc')(X)
    
    X = Dense(64, activation = 'relu', name = 'fc2')(X)
    X = Dense(7, activation = 'softmax', name = 'fc3')(X)
    
   
    
     #Block5 
    
    model = Model(inputs = X_input, outputs = X, name = 'Moodel')
    
    return model
# Starting with a high LR would break the pre-trained weights.
EPOCHS = 120
#BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU
BATCH_SIZE = 160
#LR_START = 0.00001
LR_START = 0.002
#LR_MAX = 0.00005 * tpu_strategy.num_replicas_in_sync
#LR_MAX = 0.00005
LR_MAX = 0.006
LR_MIN = 0.001
#LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 50
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .96

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
moodel1 = Moodel1(X_train.shape[1:])
moodel1.summary()
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
#lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, mode='auto')
checkpointer = ModelCheckpoint('weights.hd5', monitor='val_loss', verbose=1, save_best_only=True)
print(X_train.shape)
#history = moodel.fit(X_train,Y_train, epochs = EPOCHS, batch_size=batch_size, verbose = 1, validation_split = 0.4, shuffle=True, callbacks=[lr_reducer, checkpointer, early_stopper])
moodel1.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
history1 = moodel1.fit(X_train,Y_train, epochs = EPOCHS, batch_size=160, verbose = 1, validation_split = 0.4, shuffle=True, callbacks=[lr_callback])

moodel1.evaluate(X_test,Y_test)
final_accuracy = history1.history["val_accuracy"][-5:]
print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        #plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
display_training_curves(history1.history['accuracy'][1:], history1.history['val_accuracy'][1:], 'accuracy', 211)
display_training_curves(history1.history['loss'][1:], history1.history['val_loss'][1:], 'loss', 212)
moodel1.save('040721_4.h5')
moodel1.save('Moodelld1_5.h5')