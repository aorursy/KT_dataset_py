import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
from datetime import datetime
import  xlwt 
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imsave, imread_collection, MultiImage, imshow
%matplotlib inline
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Activation
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score, 
    classification_report, roc_auc_score, recall_score, precision_score, f1_score)
from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta, SGD, RMSprop, Adagrad, Adam, Adamax, Nadam
from keras import regularizers
from keras.models import load_model, Sequential
from keras.utils import multi_gpu_model
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras
from keras.applications import VGG16, vgg16, inception_v3, resnet50, mobilenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
import tensorflow as tf
p_lr=0.00001
p_batch_size=64
p_epochs=2
p_optimizers='SGD'
number_of_classes=3 # classe: c (Civid19), n (Normal) and p (Pneumonia) 
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    '../input/datacovid/train',
    target_size=(224, 224),
    batch_size=p_batch_size,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    '../input/datacovid/val',
    target_size=(224, 224),
    batch_size=p_batch_size,
    class_mode='categorical',
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    '../input/datacovid/test',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    shuffle=False
)
# # code by @Araújo, Flávio Apud Lima, Thiago
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="0";  # Do other imports now...
now = datetime.now()
timestamp = datetime.timestamp(now)
# # https://medium.com/@italojs/como-salvar-seus-pesos-de-%C3%A9poca-em-%C3%A9poca-keras-a9760513405e
checkpoint = ModelCheckpoint(
    "model_checkpoint_lr_"+str(p_lr)+
    "_epochs_"+str(p_epochs)+
    "_batch_size_"+str(p_batch_size)+
    "_optimizers_"+str(p_optimizers)+
    "_timestamp_"+str(timestamp)+
    "_number_of_classes_"+str(number_of_classes)+
    "_gray.h5", monitor='val_acc', verbose=1,
    save_best_only=True, mode='max', period=1
)
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0
)
#model = VGG16(weights=None, include_top=True)
#model.layers.pop()
#new_layer = Dense(number_of_classes, activation='softmax', name='predictions')
#model = Model(model.input, new_layer(model.layers[-1].output))
model = resnet50.ResNet50(weights='imagenet', include_top=True)
model.layers.pop()
new_layer = Dense(number_of_classes, activation='softmax', name='predictions')
model = Model(model.input, new_layer(model.layers[-1].output))
# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
# model = multi_gpu_model(model, gpus=[0,1])
sgd = SGD(lr=p_lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['acc'])
history = model.fit_generator(
            train_generator, 
            steps_per_epoch=train_generator.samples//train_generator.batch_size,
            epochs=p_epochs, 
            verbose=1, 
            #callbacks=[checkpoint, reduce_lr_on_plateau, EarlyStopping(monitor='loss', patience=50)],
            validation_data=val_generator,
            validation_steps=val_generator.samples//val_generator.batch_size, 
            shuffle=False
)
model.save(
    "model_lr_"+str(p_lr)+
    "_epochs_"+str(p_epochs)+
    "_batch_size_"+str(p_batch_size)+
    "_optimizers_"+str(p_optimizers)+
    "_number_of_classes_"+str(number_of_classes)+
    "_timestamp_"+str(timestamp)+
    "_gray.h5")
model.save_weights(
    'pesos_lr_'+str(p_lr)+
    '_epochs_'+str(p_epochs)+
    '_batch_size_'+str(p_batch_size)+
    '_optimizers_'+str(p_optimizers)+
    '_number_of_classes'+str(number_of_classes)+
    '_timestamp_'+ str(timestamp)+
    '_gray.hdf5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
pred = model.predict_generator(test_generator, verbose=1)
pred = np.argmax(pred, axis=1)
lb = LabelBinarizer () 

lb.fit(test_generator.labels)

y_true = lb.transform (test_generator.labels )
y_pred = lb.transform (pred)

auc_score = roc_auc_score(y_true, y_pred,average="macro")
y_test = test_generator.labels
y_pred = pred
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
kappa = cohen_kappa_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print('accuracy:', accuracy_score(y_test, y_pred))
print('recall: {}'.format(recall*100))
print('precision: {}'.format(precision*100))
print('f1-score: {}'.format(f1))
print('AUC: {}'.format(auc_score))
print('kappa: {}'.format(kappa))
print('Confusion Matriz:\n', confusion_matrix(y_test, y_pred))
loss, acc = model.evaluate_generator(test_generator, verbose=1)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
print('Restored model, loss: {:5.2f}'.format(loss))


