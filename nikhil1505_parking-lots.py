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
# Scikit-Learn libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
# Keras libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.layers import Dropout
# Initialising the CNN
classifier = Sequential()
# No. of filters is 32 with size of 3*3
# Input image size is 32*32 and 3 is for color image
# relu is the activation function in the convolution layer
classifier.add(Convolution2D(32,3,3,input_shape = (32,32,3),activation='relu'))

# Maxpooling layers is use to extract the most important feature from the feature Map(collection of filter detector)
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Convolution2D(32,3,3,activation='relu'))


classifier.add(MaxPooling2D(pool_size=(2,2)))

# Converting the matrix of size 2*2 into 1D Array
# This Flatten 1D array will be  input layer to the ANN model
classifier.add(Flatten())
import tensorflow as tf
classifier.add(Dense(output_dim = 100 , activation ='relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(output_dim = 1 , activation ='softmax'))
optimizer = SGD(lr=0.0001, momentum=0.9)
classifier.compile(optimizer =optimizer,
                   loss='binary_crossentropy',metrics=['accuracy'])
pwd
# Set the default directory
# import os 
# os.chdir("/Users/ds/Desktop/LykiQLabs/train_data")
classifier.summary()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
        '../input/parking-lots-dataset/train_data/train',
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

test_data = test_datagen.flow_from_directory(
        '../input/parking-lots-dataset/train_data/test',
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')
import pandas as pd
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
def create_stack_bar_data(col, df):
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values
train = pd.DataFrame(train_data.classes, columns=['classes'])
test = pd.DataFrame(test_data.classes, columns=['classes'])
x1, y1 = create_stack_bar_data('classes', train)
x1 = list(train_data.class_indices.keys())

trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Class Count")
layout = dict(height=600, width=600, title='Class Distribution in Training Data', legend=dict(orientation="h"), 
                yaxis = dict(title = 'Class Count'))
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);
x2, y2 = create_stack_bar_data('classes', test)
x2 = list(test_data.class_indices.keys())

Graph = go.Bar(x=x2, y=y2, opacity=0.75, name="Class Count")
layout = dict(height=600, width=600, title='Class Distribution in Validation Data', legend=dict(orientation="v"), 
                yaxis = dict(title = 'Class Count'))
fig = go.Figure(data=[Graph], layout=layout);
iplot(fig);
history_object = classifier.fit_generator(
                                        train_data,
                                        steps_per_epoch=381,
                                        epochs=10,
                                        validation_data=test_data,
                                        validation_steps=164)
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_object.history['accuracy'])
plt.plot(history_object.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
## dimensions of our images.
# Size of the image is 224*224*3 
img_width, img_height = 224, 224

train_data_dir = '../input/parking-lots-dataset/train_data/train'
validation_data_dir = '../input/parking-lots-dataset/train_data/test'
nb_train_samples = 381 
nb_validation_samples = 164
batch_size = 16
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
#import inception with pre-trained weights. do not include fully #connected layers
inception_base = applications.ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(100, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(2, activation='softmax')(x)
# create the full network so we can train on it
inception_transfer = Model(inputs=inception_base.input, outputs=predictions)


inception_base_vanilla = applications.ResNet50(weights=None, include_top=False)

# add a global spatial average pooling layer
x = inception_base_vanilla.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(200, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(2, activation='softmax')(x)
# create the full network so we can train on it
inception_transfer_vanilla = Model(inputs=inception_base_vanilla.input, outputs=predictions)
# Compiling for ResNet50 pre-trained model
inception_transfer.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
# Compiling for ResNet50 vanilla pre-trained model
inception_transfer_vanilla.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
with tf.device("/device:GPU:0"):
    history_pretrained = inception_transfer.fit_generator(
    train_generator,
    epochs=15, shuffle = True, verbose = 1, validation_data = validation_generator)
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_pretrained.history['accuracy'])
plt.plot(history_pretrained.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_pretrained.history['loss'])
plt.plot(history_pretrained.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
pwd
# Save the model according to the conditions
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
# checkpoint
filepath="/kaggle/working/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

with tf.device("/device:GPU:0"):
    history_vanilla = inception_transfer_vanilla.fit_generator(
    train_generator,
    epochs=15, shuffle = True, verbose = 1, validation_data = validation_generator, callbacks=callbacks_list)
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_vanilla.history['accuracy'])
plt.plot(history_vanilla.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_vanilla.history['loss'])
plt.plot(history_vanilla.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_pretrained.history['val_accuracy'])
plt.plot(history_vanilla.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_pretrained.history['val_loss'])
plt.plot(history_vanilla.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Reading the image
Parking_space = cv2.imread('../input/images/parking_lot1.jpg')

Parking_space = cv2.imread('../input/newline/scene1380.jpg')
Parking_space
def get_image():
    return np.copy(Parking_space)

def show_image(image):
    plt.figure(figsize=(13,12))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
show_image(get_image())
#To transfrom the colorspace from BGR to grayscale so as to make things simpler
grayimg = cv2.cvtColor(Parking_space,cv2.COLOR_BGR2GRAY)

#To plot the image
plt.figure(figsize=(13,12))
plt.imshow(grayimg,cmap='gray') #cmap has been used as matplotlib uses some default colormap to plot grayscale images

plt.xticks([]) #To get rid of the x-ticks and y-ticks on the image axis
plt.yticks([])
print('New Image Shape',grayimg.shape)
#To understand this further, let's display one entire row of the image matrix
print('The first row of the image matrix contains',len(grayimg[1]),'pixels')
print(grayimg[1])
#Okay let's look at the distribution of the intensity values of all the pixels
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(grayimg.flatten(),kde=False)#This is to flatten the matrix and put the intensity values of all the pixels in one single row vector
plt.title('Distribution of intensity values')

#To zoom in on the distribution and see if there is more than one prominent peak 
plt.subplot(1,2,2)
sns.distplot(grayimg.flatten(),kde=False) 
plt.ylim(0,30000) 
plt.title('Distribution of intensity values (Zoomed In)')
from skimage.filters import threshold_otsu
thresh_val = threshold_otsu(grayimg)
print('The optimal seperation value is',thresh_val)
mask=np.where(grayimg>thresh_val,1,0)
#To plot the original image and mask side by side
plt.figure(figsize=(13,12))
plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
maskimg = mask.copy()
plt.imshow(maskimg, cmap='rainbow')
plt.title('Mask')
#cv2.Sobel arguments - the image, output depth, order of derivative of x, order of derivative of y, kernel/filter matrix size
sobelx = cv2.Sobel(grayimg,int(cv2.CV_64F),1,0,ksize=3) #ksize=3 means we'll be using the 3x3 Sobel filter
sobely = cv2.Sobel(grayimg,int(cv2.CV_64F),0,1,ksize=3)

#To plot the vertical and horizontal edge detectors side by side
plt.figure(figsize=(13,12))
plt.subplot(1,2,1)
plt.imshow(sobelx,cmap='gray')
plt.title('Sobel X (vertical edges)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(sobely,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Sobel Y (horizontal edges)')
'''
#Plotting the original image
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='gray')
plt.title('Original image')
'''
#Now to combine the 2 sobel filters

sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
plt.figure(figsize=(13,12))
plt.subplot(1,1,1)
plt.imshow(sobel,cmap='gray')
plt.title('Sobel Filter')
#To highlight the problem areas
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(grayimg[348:360,485:521],cmap='gray')
plt.title('Original image (zoomed in)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(sobel[348:360,485:521],cmap='gray')
plt.title('Sobel Filter (zoomed in)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(maskimg[348:360,485:521], cmap='gray')
plt.title('Otsu/K-Means (zoomed in)')
plt.xticks([])
plt.yticks([])

#To highlight the problem areas
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(grayimg[345:445,488:537],cmap='gray')
plt.title('Original image (zoomed in)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(sobel[345:445,488:537],cmap='gray')
plt.title('Sobel Filter (zoomed in)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(maskimg[345:445,488:537], cmap='gray')
plt.title('Otsu/K-Means (zoomed in)')
plt.xticks([])
plt.yticks([])

 

