#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
!ls ../lib/kaggle
#@title MIT License
#
# Copyright (c) 2017 François Chollet                                                                                                                    # IGNORE_COPYRIGHT: cleared by OSS licensing
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2



print("TensorFlow version is ", tf.__version__)

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
!ls ../input
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')

# # Directory with our training cat pictures
# train_cats_dir = os.path.join(train_dir, 'cats')
# print ('Total training cat images:', len(os.listdir(train_cats_dir)))

# # Directory with our training dog pictures
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# print ('Total training dog images:', len(os.listdir(train_dogs_dir)))

# # Directory with our validation cat pictures
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# print ('Total validation cat images:', len(os.listdir(validation_cats_dir)))

# # Directory with our validation dog pictures
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# print ('Total validation dog images:', len(os.listdir(validation_dogs_dir)))
train_dir = '../input/tiny_imagenet_flipped/train'
validation_dir = '../input/tiny_imagenet_flipped/val'
image_size = 224 # All images will be resized to 160x160
train_batchsize = 64
val_batchsize = 64
IMAGE_SIZE = (image_size, image_size)

# Rescale all images by 1./255 and apply image augmentation
datagen_kwargs = dict(rescale=1./255)
dataflow_kwargs = dict(target_size=IMAGE_SIZE,
                   interpolation="bilinear", class_mode='categorical')

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
validation_generator = valid_datagen.flow_from_directory(
    validation_dir, shuffle=False, batch_size=val_batchsize, **dataflow_kwargs)

do_data_augmentation = True #@param {type:"boolean"}
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,fill_mode='nearest',
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    train_dir, shuffle=True, batch_size=train_batchsize, **dataflow_kwargs)
IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model Mobilenetv2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
len(base_model.layers)
base_model.output
# Train the entire model
base_model.trainable = True
fine_tune_at = 55

# # Freeze the layers except the last 100 layers
# for layer in base_model.layers[:fine_tune_at]:
#    layer.trainable =  False

 
# Check the trainable status of the individual layers
for i, layer in enumerate(base_model.layers):
    print(i, layer, layer.trainable)
# Let's take a look at the base model architecture
base_model.summary()
num_classes=4
model = keras.models.Sequential()
 
# Add the vgg convolutional base model
model.add(base_model)
# Add new layers
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(1280, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.Dense(640, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Dense(150, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()


model.compile(optimizer=tf.keras.optimizers.RMSprop(2e-3, decay=1e-4, momentum=0.85),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)
# Implement One Cycle Policy Algorithm in the Keras Callback Class

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
import tensorflow

class CyclicLR(tensorflow.keras.callbacks.Callback):
    
    def __init__(self,base_lr, max_lr, step_size, base_m, max_m, cyclical_momentum):
 
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.base_m = base_m
        self.max_m = max_m
        self.cyclical_momentum = cyclical_momentum
        self.step_size = step_size
        
        self.clr_iterations = 0.
        self.cm_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        
    def clr(self):
        
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        
        if cycle == 2:
            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)          
            return self.base_lr-(self.base_lr-self.base_lr/100)*np.maximum(0,(1-x))
        
        else:
            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0,(1-x))
    
    def cm(self):
        
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        
        if cycle == 2:
            
            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1) 
            return self.max_m
        
        else:
            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
            return self.max_m - (self.max_m-self.base_m)*np.maximum(0,(1-x))
        
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())
            
        if self.cyclical_momentum == True:
            if self.clr_iterations == 0:
                K.set_value(self.model.optimizer.momentum, self.cm())
            else:
                K.set_value(self.model.optimizer.momentum, self.cm())
            
            
    def on_batch_begin(self, batch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        if self.cyclical_momentum == True:
            self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
        
        if self.cyclical_momentum == True:
            K.set_value(self.model.optimizer.momentum, self.cm())

epochs = 50
steps_per_epoch = train_generator.n // train_generator.batch_size 
validation_steps = validation_generator.n // validation_generator.batch_size

epochs = 10
max_lr = 0.003
base_lr = max_lr/10
max_m = 0.95
base_m = 0.85

cyclical_momentum = True
augment = True
cycles = 2.35

iterations = round(train_generator.n/batch_size*epochs)
iterations = list(range(0,iterations+1))
step_size = len(iterations)/(cycles)


model.compile(optimizer=tf.keras.optimizers.SGD(decay=1e-4,),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

clr =  CyclicLR(base_lr=base_lr,
                max_lr=max_lr,
                step_size=step_size,
                max_m=max_m,
                base_m=base_m,
                cyclical_momentum=cyclical_momentum)

early_stopping_callback = EarlyStopping(monitor='val_acc', patience=30)

checkpoint_callback = ModelCheckpoint(
                                     'classifier_mobilenetv2_model_weights_aug.h5', 
                                      save_best_only=True, 
                                      monitor='val_acc', verbose=1, 
                                      save_weights_only=True, mode='auto')

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              callbacks=[early_stopping_callback, checkpoint_callback, clr],
                              validation_steps=validation_steps)
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()
# Create a generator for prediction
datagen_kwargs = dict(rescale=1./255)
dataflow_kwargs = dict(target_size=IMAGE_SIZE,
                   interpolation="bilinear", class_mode='categorical')

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
validation_generator = valid_datagen.flow_from_directory(
    validation_dir, shuffle=False, batch_size=val_batchsize, **dataflow_kwargs)
 
# Get the filenames from the generator
fnames = validation_generator.filenames
 
# Get the ground truth from generator
ground_truth = validation_generator.classes
 
# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
 
# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
     
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])
     
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
test_dir = '../inptest'
datagen_kwargs = dict(rescale=1./255)
dataflow_kwargs = dict(target_size=IMAGE_SIZE,
                   interpolation="bilinear", class_mode='categorical')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
test_generator = test_datagen.flow_from_directory(
    test_dir, shuffle=False, batch_size=val_batchsize, **dataflow_kwargs)
loss, acc = model.evaluate_generator(test_generator ,steps = test_generator.n//val_batchsize)
acc