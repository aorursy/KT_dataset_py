import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
from keras.utils.io_utils import HDF5Matrix
train_img = HDF5Matrix(os.path.join('..', 'input', 'scans_train.h5'), 'image')[:][:, ::4, ::4]/255.0

train_label = HDF5Matrix(os.path.join('..', 'input', 'scans_train.h5'), 'Region')



valid_img = HDF5Matrix(os.path.join('..', 'input', 'scans_valid.h5'), 'image')[:][:, ::4, ::4]/255.0

valid_label = HDF5Matrix(os.path.join('..', 'input', 'scans_valid.h5'), 'Region')

print('Training:', train_img.shape, train_label.shape)

print('Validation:', valid_img.shape, valid_label.shape)
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

lab_enc = LabelEncoder()

lab_enc.fit(train_label[:])

train_y = to_categorical(lab_enc.transform(train_label[:]))

valid_y = to_categorical(lab_enc.transform(valid_label[:]))

print('Region Names', lab_enc.classes_)

print('Y output', train_y.shape, valid_y.shape)
from keras.applications.mobilenet import MobileNet

s_net = MobileNet(classes=valid_y.shape[1], 

                  weights = None, 

                  input_shape=train_img.shape[1:],

                 dropout = 0.5)

s_net.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])

print('Layers: {}, parameters: {}'.format(len(s_net.layers), s_net.count_params()))
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

file_path="weights.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_acc", mode="max", patience=3)

callbacks_list = [checkpoint, early] #early
s_net.fit(train_img, train_y, batch_size = 128,

          validation_data = (valid_img, valid_y),

          epochs=2,

          callbacks = callbacks_list)
try:

    s_net.load_weights(file_path) # load the best model

except:

    pass # no file found

pred_y = s_net.predict(valid_img, verbose = True, batch_size = 256)

pred_cat_y = np.argmax(pred_y, -1)

pred_conf_y = pred_y[:,1]

out_cat_y = np.argmax(valid_y, -1)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(out_cat_y, pred_cat_y))

fig, ax1 = plt.subplots(1,1,figsize = (5,5))

ax1.matshow(confusion_matrix(out_cat_y, pred_cat_y))

ax1.set_xlabel('Model Prediction')

ax1.set_ylabel('Actual Label')

ax1.set_yticklabels(['']+[x.decode() for x in lab_enc.classes_])

print(lab_enc.classes_)