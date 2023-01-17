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
import pandas as pd
import numpy as np
from matplotlib import pyplot
import scipy.misc 
from math import sqrt 
import itertools
from IPython.display import display
%matplotlib inline
#Loading the dataset
data= pd.read_csv('../input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')
data.head()
#Number of training samples of each emotions
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
#Assigning Names to Emotions in labels

num_classes = 7
width = 48
height = 48
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
classes=np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))
data.Usage.value_counts() 
depth = 1
height = int(sqrt(len(data.pixels[0].split()))) 
width = int(height)
for i in range(0, 10): 
    array = np.mat(data.pixels[i]).reshape(height, width) 
    image = scipy.misc.toimage(array, cmin=0.0) 
    display(image)
    print(emotion_labels[data.emotion[i]]) 
train_set = data[(data.Usage == 'Training')] 
val_set = data[(data.Usage == 'PublicTest')]
test_set = data[(data.Usage == 'PrivateTest')] 
X_train = np.array(list(map(str.split, train_set.pixels)), np.float32) 
X_val = np.array(list(map(str.split, val_set.pixels)), np.float32) 
X_test = np.array(list(map(str.split, test_set.pixels)), np.float32) 
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) 
X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
num_train = X_train.shape[0]
num_val = X_val.shape[0]
num_test = X_test.shape[0]
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator 
y_train = train_set.emotion 
y_train = np_utils.to_categorical(y_train, num_classes) 
y_val = val_set.emotion 
y_val = np_utils.to_categorical(y_val, num_classes) 
y_test = test_set.emotion 
y_test = np_utils.to_categorical(y_test, num_classes)
datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')

testgen = ImageDataGenerator( 
    rescale=1./255
    )
datagen.fit(X_train)
batch_size = 64
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
    for i in range(0, 9): 
        pyplot.axis('off') 
        pyplot.subplot(330 + 1 + i) 
        pyplot.imshow(X_batch[i].reshape(48, 48), cmap=pyplot.get_cmap('gray'))
    pyplot.axis('off') 
    pyplot.show() 
    break
train_flow = datagen.flow(X_train, y_train, batch_size=batch_size) 
val_flow = testgen.flow(X_val, y_val, batch_size=batch_size) 
test_flow = testgen.flow(X_test, y_test, batch_size=batch_size)
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
# def FER_Model(input_shape=(48,48,1)):
#     # first input model
#     visible = Input(shape=input_shape, name='input')
#     num_classes = 7
#     #the 1-st block
#     conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
#     conv1_1 = BatchNormalization()(conv1_1)
#     conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
#     conv1_2 = BatchNormalization()(conv1_2)
#     pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
#     drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)

#     #the 2-nd block
#     conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
#     conv2_1 = BatchNormalization()(conv2_1)
#     conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
#     conv2_2 = BatchNormalization()(conv2_2)
#     conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
#     conv2_2 = BatchNormalization()(conv2_3)
#     pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
#     drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)

#      #the 3-rd block
#     conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
#     conv3_1 = BatchNormalization()(conv3_1)
#     conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
#     conv3_2 = BatchNormalization()(conv3_2)
#     conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
#     conv3_3 = BatchNormalization()(conv3_3)
#     conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
#     conv3_4 = BatchNormalization()(conv3_4)
#     pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
#     drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)

#     #the 4-th block
#     conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
#     conv4_1 = BatchNormalization()(conv4_1)
#     conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
#     conv4_2 = BatchNormalization()(conv4_2)
#     conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
#     conv4_3 = BatchNormalization()(conv4_3)
#     conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
#     conv4_4 = BatchNormalization()(conv4_4)
#     pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
#     drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)
    
#     #the 5-th block
#     conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
#     conv5_1 = BatchNormalization()(conv5_1)
#     conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
#     conv5_2 = BatchNormalization()(conv5_2)
#     conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
#     conv5_3 = BatchNormalization()(conv5_3)
#     conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
#     conv5_3 = BatchNormalization()(conv5_3)
#     pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
#     drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)

#     #Flatten and output
#     flatten = Flatten(name = 'flatten')(drop5_1)
#     ouput = Dense(num_classes, activation='sigmoid', name = 'output')(flatten)

#     # create model 
#     model = Model(inputs =visible, outputs = ouput)
#     # summary layers
#     print(model.summary())
    
#     return model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.optimizers import Adam
import cv2, numpy as np
def FER_Model(input_shape=(48,48,1)):
    # first input model
    visible = Input(shape=input_shape, name='input')
    num_classes = 7
    #the 1-st block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)

    #the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)
    #the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)
  #the 4-th block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
    drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)
    #the 5-th block
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
    conv5_3 = BatchNormalization()(conv5_3)
    pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
    drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)

    #Flatten and output
    flatten = Flatten(name = 'flatten')(drop5_1)
    ouput = Dense(num_classes, activation='softmax', name = 'output')(flatten)

    # create model 
    model = Model(inputs =visible, outputs = ouput)
    # summary layers
    print(model.summary())
    return model
# def VGG_16():
#     model = Sequential()
#     model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
#     model.add(Conv2D(64,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(64,(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
# #     model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(128,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(128,(3, 3), activation='relu'))
#   #  model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#     model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256,(3, 3), activation='relu'))
# #    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#     model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#   #  model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#  #   model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(Flatten())
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))
#     return model

# def VGG_19():
#     model = Sequential()
#     model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
#     model.add(Conv2D(64,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(64,(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
# #     model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(128,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(128,(3, 3), activation='relu'))
#   #  model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#     model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256,(3, 3), activation='relu'))
# #    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#     model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#   #  model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512,(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),dim_ordering="th"))
#  #   model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

#     model.add(Flatten())
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))
#     return model

model = FER_Model()
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# opt = Adam(lr=0.0001, decay=1e-6)
opt = Adam(lr=0.0001, decay=1e-6)
opt = Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

# model = FER_Model()
# opt = Adam(lr=0.0001, decay=1e-6)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
filepath="weights_min_loss.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# we iterate 200 times over the entire training set
num_epochs = 50
# history = model.fit_generator(train_flow, 
#                     steps_per_epoch=len(X_train) / batch_size, 
#                     epochs=num_epochs,  
#                     verbose=2,  
#                     callbacks=callbacks_list,
#                     validation_data=val_flow,  
#                     validation_steps=len(X_val) / batch_size)
history = model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(np.array(X_val), np.array(y_val)),
          shuffle=True)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save('../working/Fer2013_VGG19adam0005.h5')
def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title='Unnormalized confusion matrix',
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_test, y_pred)
    
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        
    np.set_printoptions(precision=2)
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.min() + (cm.max() - cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True expression')
    plt.xlabel('Predicted expression')
    plt.show()
y_pred_ = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_, axis=1)
t_te = np.argmax(y_test, axis=1)
fig = plot_confusion_matrix(y_test=t_te, y_pred=y_pred,
                      classes=classes,
                      normalize=True,
                      cmap=plt.cm.Greys,
                      title='Average accuracy: ' + str(np.sum(y_pred == t_te)/len(t_te)) + '\n')