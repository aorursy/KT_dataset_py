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
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import numpy as np
from PIL import Image
from math import floor
import numpy as np
import time
from functools import partial
from random import random
from pandas import Series, DataFrame
import pandas as pd
import seaborn as sns
import pickle
from keras.optimizers import SGD
import cv2
import os
import keras
from keras import layers
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dropout
from keras.models import Model,Sequential
from keras.layers import Input,MaxPooling2D,concatenate,Dense,Flatten
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
#from tensorflow.python.keras.layers import Dense
#from random_eraser import get_random_eraser
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Activation,Conv2D,Flatten,SeparableConv2D,Dense,Input,Dropout,BatchNormalization,GlobalMaxPooling2D,GlobalAveragePooling2D,MaxPooling2D,AveragePooling2D
os.listdir('../input/')
directory_root = '../input/'
test_dir='../input/testdata/'
batch_size=32
train_data_dir='../input/dancedata/Train Images/'
#train_data_dir='../input/newdata/stage 02/'
num_classes=2
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

#test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    batch_size=batch_size,
    target_size=(224,224),
    class_mode='categorical',
    subset='training') # set as training data
train_generator.class_indices
validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    batch_size=batch_size,
    target_size=(224,224),
    class_mode='categorical',
    subset='validation')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir, # same directory as training data
    batch_size=batch_size,
    target_size=(224,224),
    class_mode='categorical')
#mobilenetsv2
base_model=MobileNetV2(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dropout(0.1)(x)
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dropout(0.1)(x)
x=Dense(512,activation='relu')(x) #dense layer 3
#preds=Dense(2,activation='sigmoid')(x) #final layer with softmax activation
preds=Dense(8,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True
#mobilenetsv2
base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
base_model.trainable=False
#VGG16
model = Sequential()
base_model=VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))

model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))
base_model.trainable=False
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#inceptionv3
nclass=8
import os
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop

# local_weights_file = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (224,224, 3), 
                                include_top = False, 
                                weights = "imagenet")

# pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
     layer.trainable = False
        
# pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(8, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 
model.summary()
#model.compile(optimizer = RMSprop(lr=0.0001),loss = 'sparse_categorical_crossentropy',metrics = ['acc'])
#model.summary()
#history=model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))


#base_model = InceptionV3(weights='imagenet', include_top=False)
#base_model.trainable = False
#model = Sequential()
#model.add(base_model)
#model.add(GlobalAveragePooling2D())
#model.add(Dropout(0.5))
#model.add(Dense(nclass, activation='softmax'))

#model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (224,224, 3))
base_model.trainable = False
model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:

#model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
#resnet50
num_classes=8
base_model = Sequential()
base_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet')) 	# entire resnet model is the first layer!
base_model.add(Dense(num_classes, activation='softmax'))
base_model.layers[0].trainable = False
#base_model.summary()
#base_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#achieved acc of- 
#resnet50
model = ResNet50(weights='imagenet',include_top=False)
model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()

for layer in custom_resnet_model2.layers[:-6]:
	layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)
es1 = EarlyStopping(monitor='accuracy', mode='max', verbose=1,patience=15)
callbacks_list = [es,reduce_learning_rate,es1]
model=base_model
INIT_LR = 1e-3
#opt = Adam(lr=INIT_LR, decay=INIT_LR / 50)
#opt='rmsprop'
#opt='adam'
#opt=SGD(lr=1e-4, momentum=0.9)
#opt = SGD(lr=0.001, momentum=0.9)
opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#opt=Adam()
# distribution
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.compile(optimizer=opt, loss='binary_crossentropy',  metrics=['accuracy'])
# train the network

print("[INFO] training network...")
model=base_model
history=model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = 30,shuffle=True,
    #callbacks=callbacks_list
    )
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")    
from IPython.display import FileLink
FileLink(r'model.h5')
#evaluate_model
#STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.evaluate_generator(generator=validation_generator,
steps=STEP_SIZE_VALID)
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size +1
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size +1
#test_generator.reset()
y_pred = model.predict_generator(validation_generator, steps=STEP_SIZE_VALID,
verbose=1)
y_true=validation_generator.classes
#print(len(y_true))
#print(y_true)
#train_data_prediction
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size +1
#test_generator.reset()
y_pred = model.predict_generator(train_generator, steps=STEP_SIZE_TRAIN,
verbose=1)
y_true=train_generator.classes
#print(len(y_true))
#print(y_true)
predicted_class_indices=np.argmax(y_pred,axis=1)
print(predicted_class_indices)
print(len(predicted_class_indices))
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
#validation_data_prediction
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size +1
#test_generator.reset()
y_pred = model.predict_generator(validation_generator, steps=STEP_SIZE_VALID,
verbose=1)
y_true=validation_generator.classes
#print(len(y_true))
#print(y_true)
predicted_class_indices=np.argmax(y_pred,axis=1)
print(predicted_class_indices)
print(len(predicted_class_indices))
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
#test_data_prediction
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size +1
#test_generator.reset()
y_pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST,verbose=1)
#y_true=test_generator.classes
#print(len(y_true))
#print(y_true)
predicted_class_indices=np.argmax(y_pred,axis=1)
print(predicted_class_indices)
print(len(predicted_class_indices))
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
tn, fp, fn, tp=confusion_matrix(y_true, predicted_class_indices).ravel()
print(confusion_matrix(y_true, predicted_class_indices))
print((tp+tn)/(tp+tn+fp+fn))
print('tn, fp, fn, tp')
print((tn, fp, fn, tp))
import pandas as pd
filenames=test_generator.filenames
y=[]
#print(len(filenames))
for i in range(len(filenames)):
    a,b=filenames[i].split('/')
    y.append(b)
#print(y)
results=pd.DataFrame({"Image":y,
                      "target":predictions})
results.to_csv("results.csv",index=False)
from IPython.display import FileLink
FileLink(r'results.csv')