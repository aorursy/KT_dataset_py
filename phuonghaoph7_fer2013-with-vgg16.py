from keras.utils import to_categorical
import tensorflow as tf
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import seaborn as sns
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import*
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from keras.applications import VGG16;
from keras.applications.vgg16 import preprocess_input
import tensorflow_hub as tfHub
from tensorflow.keras import layers as tfLayers
from keras import models
from keras.models import Model
from keras import layers
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout,Flatten,BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import adam
import os
train_0=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train/0')
train_1=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train/1')
train_2=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train/2')
train_3=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train/3')
train_4=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train/4')
train_5=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train/5')
train_6=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train/6')

test_0=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test/0')
test_1=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test/1')
test_2=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test/2')
test_3=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test/3')
test_4=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test/4')
test_5=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test/5')
test_6=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test/6')

val_0=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val/0')
val_1=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val/1')
val_2=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val/2')
val_3=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val/3')
val_4=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val/4')
val_5=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val/5')
val_6=os.listdir('/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val/6')

n_train=len(train_1)+len(train_2)+len(train_3)+len(train_4)+len(train_5)+len(train_6)+len(train_0)
n_test=len(test_1)+len(test_2)+len(test_3)+len(test_4)+len(test_5)+len(test_6)+len(test_0)
n_val=len(val_1)+len(val_2)+len(val_3)+len(val_4)+len(val_5)+len(val_6)+len(val_0)
print('So anh tap train: ',n_train, '  class0:',len(train_0),'     class1:',len(train_1),'      class2:',len(train_2), '    class3:',len(train_3),'     class4:',len(train_4),'        class5:',len(train_5),'     class6:',len(train_6))
print('So anh tap test: ',n_test, '    class0:',len(test_0),'      class1:',len(test_1),'       class2:',len(test_2), '    class3:',len(test_3),'        class4:',len(test_4),'        class5:',len(test_5),'      class6:',len(test_6))
print('So anh tap val: ',n_val, '      class0:',len(val_0),'       class1:',len(val_1),'        class2:',len(val_2), '    class3:',len(val_3),'        class4:',len(val_4),'           class5:',len(val_5),'       class6:',len(val_6))
batch_size = 64
width = 48
height = 48
train_dir = '/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/train'
test_dir =  '/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/test'
val_dir ='/content/my_data/face_emotion-20200711T091042Z-001/face_emotion/val'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      featurewise_center=False,
      featurewise_std_normalization=False,
      rotation_range=10,
      width_shift_range=0.1,
      height_shift_range=0.1,
      zoom_range=.1,
      horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to target height and width.
        target_size=(height, width),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint

class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes), 
        train_generator.classes)
num_classes=7
keys = range(num_classes)
values = class_weights.copy()
class_weights = dict(zip(keys, values))
print(class_weights)
from keras.applications import VGG16;
base_model = VGG16(
    include_top = False,
    weights     = 'imagenet',
    input_shape = (48,48, 3))

for layer in base_model.layers[:10]:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x=Dense(256,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(7,activation='softmax')(x)
#,kernel_regularizer='l2'
model = Model(inputs=base_model.input, outputs=x)

opt = SGD(lr=0.01)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc'])

history = model.fit_generator(
    train_generator,   
    epochs=70,
    verbose=1,  
    class_weight=class_weights,
    validation_data=val_generator,
    

)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224,224,3),
        batch_size=64,
        )

pred = model.predict_generator(test_generator)
pred = np.argmax(pred, axis=-1)
y_true = test_generator.classes

d=0
for i in range (0,len(y_true)):
  if(y_test[i]==pred[i]):
    d=d+1
acc= (d/len(y_test))*100.0
print(acc)

from sklearn.metrics import classification_report

print(classification_report((y_true), pred))