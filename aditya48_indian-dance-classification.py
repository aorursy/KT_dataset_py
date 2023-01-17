import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from sklearn.metrics import confusion_matrix
# np.random.seed(2)

from keras.utils.np_utils import to_categorical

import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
train_path = "../input/indian-dance-form-classification/train/"
test_path = "../input/indian-dance-form-classification/test/"

kathak = "../input/indian-dance-form-classification/train/kathak/"
odissi = "../input/indian-dance-form-classification/train/odissi/"
sattriya = "../input/indian-dance-form-classification/train/sattriya/"

kathak_path = os.listdir(kathak)
sattriya_path = os.listdir(sattriya)
odissi_path = os.listdir(odissi)
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
#     print(labels)
    return image[...,::-1]


plt.imshow(load_img(kathak + kathak_path[2]), cmap='gray')
plt.imshow(load_img(odissi + odissi_path[2]), cmap='gray')
plt.imshow(load_img(sattriya + sattriya_path[2]), cmap='gray')
training_data = []
IMG_SIZE = 224

datadir = "../input/indian-dance-form-classification/train/"


categories = ['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi',  'manipuri', 'mohiniyattam', 'odissi', 'sattriya']

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except:
                pass
create_training_data()
training_data = np.array(training_data)
print(training_data.shape)
import random

np.random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)
print(X.shape)
print(y.shape)
print(np.unique(y, return_counts = True))

print(y[1:10])
a,b = np.unique(y, return_counts = True)
print(a)
print(b)
print(categories)
import plotly.graph_objs as go 
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

trace = go.Bar(x = categories, y = b)
data = [trace]
layout = {"title":"Categories vs Images Distribution",
         "xaxis":{"title":"Categories","tickangle":0},
         "yaxis":{"title":"Number of Images"}}
fig = go.Figure(data = data,layout=layout)
iplot(fig)
X = X/255.0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)
print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)
y_train = to_categorical(y_train, num_classes = 8)
y_test = to_categorical(y_test, num_classes = 8)
print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)
print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

train_x = tf.keras.utils.normalize(X_train,axis=1)
test_x = tf.keras.utils.normalize(X_test, axis=1)
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout,Activation
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

# For interupt the training when val loss is stagnant
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
model = keras.applications.VGG16(input_shape = (224,224,3), weights = 'imagenet',include_top=False)

for layer in model.layers:
    layer.trainable = False

last_layer = model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)

# add fully-connected & dropout layers
x = Dense(4096, activation='relu',name='fc-1')(x)
x = Dropout(0.2)(x)
x = Dense(4096, activation='relu',name='fc-2')(x)
x = Dropout(0.2)(x)

# x = Dense(4096, activation='relu',name='fc-3')(x)
# x = Dropout(0.2)(x)

# a softmax layer for 8 classes
num_classes = 8
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
model2 = Model(inputs=model.input, outputs=out)

model2.summary()
model2.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=3)


# hist = model2.fit(X_train,y_train, batch_size=30, epochs = 100, validation_data = (X_test,y_test), callbacks=[early_stopping])
hist = model2.fit(X_train,y_train, batch_size=30, epochs = 30, validation_data = (X_test,y_test))
# Visualizing the training. 

epochs = 30

# The uncomment everything in this cell and run it.

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout,Activation
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

# For interupt the training when val loss is stagnant
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
model = keras.applications.VGG19(input_shape = (224,224,3), weights = 'imagenet',include_top=False)

for layer in model.layers:
    layer.trainable = False

last_layer = model.output
# add a global spatial average pooling layer


x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(4096, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu',name='fc-3')(x)
# x = Dropout(0.5)(x)

# a softmax layer for 8 classes
num_classes = 8
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
model2 = Model(inputs=model.input, outputs=out)

model2.summary()
model2.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=4)


hist = model2.fit(X_train,y_train, batch_size=30, epochs = 15, validation_data = (X_test,y_test))
## Uncomment everything once you find the number of epochs.

epochs = 15 # should be equal to the number of epochs that the training had took place.
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout,Activation
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
## Just replace Xception with ResNet50 and ResNet101 for trying these models. Honestly both models 
## performed poorly on this dataset.

model = keras.applications.Xception(input_shape = (224,224,3), weights = 'imagenet',include_top=False)
model.summary()

for layer in model.layers:
	layer.trainable = False

last_layer = model.output
# add a global spatial average pooling layer


x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers

x = Dense(4096, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu',name='fc-2')(x)
# x = Dropout(0.2)(x)


# a softmax layer for 8 classes
num_classes = 8
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
model2 = Model(inputs=model.input, outputs=out)

# model2.summary()


model2.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

hist = model2.fit(X_train,y_train, batch_size=10, epochs = 20, validation_data = (X_test, y_test))
epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
# test_data = []
# img_id = []
# IMG_SIZE = 224


# def create_testing_data():
#     path = "../input/indian-dance-form-classification/test/"
#     for img in os.listdir(path):
#         try:
#             img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
#             new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#             test_data.append([new_array])
#             img_id.append([img])
#         except:
#             pass
# create_testing_data()
# print(len(test_data))
# test_data = np.array(test_data)
# test_data =  test_data.reshape(156,224,224,3)
# print(test_data.shape)
# test_data = test_data/255

# ## ID of Images.
# print(img_id[0])
# import pandas as pd
# image = []
# for i in img_id:
#     image.append(i[0])

# image = np.array(image)
# print(image.shape)
# test = pd.DataFrame(image, columns = ['Image'])
# test.head()
# import pandas as pd

# predict = model2.predict(test_data)
# predict = np.argmax(predict,axis = 1)
# predict = np.array(predict)
# print(predict.shape)

# print(np.unique(predict,return_counts = True))

# x = ['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi',  'manipuri', 'mohiniyattam','odissi','sattriya']
# y = []
# for i in predict:
#     y.append(x[i])

# print(y)
# y = np.array(y)
# pred = pd.DataFrame(y, columns = ['target'])
# pred.head()


# test_csv = pd.concat([test,pred], axis=1)
# # df.sort_values(by=['col1'])
# new = test_csv.sort_values(by=['Image'])
# print(new.head())

# new.to_csv("test.csv",columns = list(test_csv.columns),index=False)
