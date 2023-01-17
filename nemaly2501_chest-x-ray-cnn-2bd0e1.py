import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
%matplotlib inline
import matplotlib.pyplot as plt
import h5py
from keras.utils.io_utils import HDF5Matrix
h5_path = '../input/create-a-mini-xray-dataset-equalized/chest_xray.h5'
disease_vec_labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis',
 'Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
disease_vec = []
with h5py.File(h5_path, 'r') as h5_data:
    all_fields = list(h5_data.keys())
    for c_key in all_fields:
        print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)
    for c_key in disease_vec_labels:
        disease_vec += [h5_data[c_key][:]]
disease_vec = np.stack(disease_vec,1)
print('Disease Vec:', disease_vec.shape)
img_ds = HDF5Matrix(h5_path, 'images')
split_idx = img_ds.shape[0]//2
train_ds = HDF5Matrix(h5_path, 'images', end = split_idx)
test_ds = HDF5Matrix(h5_path, 'images', start = split_idx)
train_dvec = disease_vec[0:split_idx]
test_dvec = disease_vec[split_idx:]
print('Train Shape', train_ds.shape, 'test shape', test_ds.shape)
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
raw_model = MobileNet(input_shape=(None, None, 1), include_top = False, weights = None)
full_model = Sequential()
full_model.add(AveragePooling2D((2,2), input_shape = img_ds.shape[1:]))
full_model.add(BatchNormalization())
full_model.add(raw_model)
full_model.add(Flatten())
full_model.add(Dropout(0.5))
full_model.add(Dense(64))
full_model.add(Dense(disease_vec.shape[1], activation = 'sigmoid'))
full_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
full_model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
file_path="weights.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=3)
callbacks_list = [checkpoint, early] #early
full_model.fit(train_ds, train_dvec, 
               validation_data = (test_ds, test_dvec),
               epochs=5, 
               verbose = True,
              shuffle = 'batch',
              callbacks = callbacks_list)
x = full_model.predict(test_ds[:200])
x.tolist()
test_dvec[:3]
import matplotlib.pyplot as plt
for i in range(50):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.text(0,0,i)
    ax.tick_params(axis='x', labelrotation=45)
    ax.bar(disease_vec_labels,x[i], label = i)
    plt.show()
for i in range(50):    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.text(0,0,i)
    ax.tick_params(axis='x', labelrotation=45)
    ax.bar(disease_vec_labels,test_dvec[i])
    plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0,0.005,3)
ax.tick_params(axis='x', labelrotation=45)
ax.bar(disease_vec_labels,x[42], label = 42)
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0,0.5,i)
ax.tick_params(axis='x', labelrotation=45)
ax.bar(disease_vec_labels,test_dvec[42], label = 42)
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.text(0,0.5,i)
ax.tick_params(axis='x', labelrotation=45)
ax.bar(disease_vec_labels,test_dvec[42], label = 42)
plt.show()
full_model.summary()
full_model.save('model.h5')
full_model.save_weights

from keras.models import load_model
model = load_model('model.h5')