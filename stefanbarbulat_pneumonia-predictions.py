# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import h5py

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
folder = glob('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/*')
folder = glob('../input/chest-xray-pneumonia/chest_xray/train/*')
img = cv2.imread('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/PNEUMONIA/SARS-10.1148rg.242035193-g04mr34g0-Fig8c-day10.jpeg')
plt.imshow(img, cmap=plt.cm.bone)
IMAGE_SIZE = [224,224]
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
preds = Dense(512, activation='relu')(x)
predictions = Dense(len(folder), activation='softmax')(preds)

vgg_model = Model(inputs=vgg.input, outputs=predictions)
vgg_model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=[categorical_accuracy])
test_datagen = ImageDataGenerator(rescale = 1./255)
#For train data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_set = train_datagen.flow_from_directory('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/', 
                                                            target_size = (224, 224),
                                                                batch_size = 32, 
                                                            class_mode = 'categorical'
                                                )
train_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train/', 
                                                            target_size = (224, 224),
                                                                batch_size = 32, 
                                                            class_mode = 'categorical'
                                                )

# for test data
test_set = test_datagen.flow_from_directory('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical'
                                           )
filepath = "vgg16_model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = vgg_model.fit_generator(train_set, steps_per_epoch=len(train_set), 
                              #class_weight=class_weights,
                    validation_data=test_set,
                    validation_steps=len(test_set),
                    epochs=1, verbose=1,
                   callbacks=callbacks_list)
y_pred = vgg_model.predict_generator(test_set)
y_pred = np.argmax(y_pred, axis=1)
y_pred
test_set.class_indices
batch_size = 32
val_steps = np.ceil(len(test_set) / batch_size)

val_loss, val_cat_acc = \
vgg_model.evaluate_generator(test_set, 
                        steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
vgg_model.load_weights('vgg16_model.h5')

val_loss, val_cat_acc = \
vgg_model.evaluate_generator(test_set, 
                        steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(test_set.classes,y_pred)
cm
tn, fp, fn, tp = cm.ravel()
print('False Positives : {}'.format(fp))
print('False Negatives: {}'. format(fn))
print('True Positives : {}'.format(tp))
sns.heatmap(cm, annot=True, cmap='coolwarm')
print('The Classification Report : \n{}'.format(classification_report(test_set.classes,y_pred)))
from skimage import io
from keras.preprocessing import image
#path='imbalanced/Scratch/Scratch_400.jpg'
img = image.load_img('../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19 (125).png', grayscale=False, target_size=(224, 224))
show_img=image.load_img('../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19 (125).png', grayscale=False, target_size=(224, 224))
disease_class=['Normal','Pneumonie']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

custom = vgg_model.predict(x)
print(custom[0])

plt.imshow(show_img)
plt.show()

a=custom[0]
ind=np.argmax(a)
        
print('Prediction:',disease_class[ind])
# End of Model Building
### ===================================================================================== ###
# Convert the Model from Keras to Tensorflow.js
!pip install tensorflowjs
!pip install --upgrade pip