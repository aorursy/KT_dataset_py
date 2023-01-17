import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import math
import h5py
import random
import nilearn as nl
from nilearn import image, datasets, plotting
from nilearn.image import get_data
from random import randint
from tqdm.notebook import tqdm
from sklearn.impute import KNNImputer
from tqdm.notebook import tqdm
source = "../input/trends-assessment-prediction"
img_train = []
for name in os.listdir(source+'/fMRI_train'):
    img_train.append(int(name[:5]))
img_train.sort()
train = {}
for i, name in enumerate(img_train):
    train[i] = name
impute = KNNImputer(n_neighbors=40)
y_train = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")
y_train = impute.fit_transform(y_train)
y_train = pd.DataFrame(y_train)
y_train
y = {}
for i in range(len(img_train)):
    y[img_train[i]] = y_train[y_train[0] == img_train[i]].drop([0], axis=1).to_numpy()
def get_batch(k, size):
    image = img_train[k:k+size]
    batch = np.zeros((size,53, 52, 63, 53))
    y_train = np.zeros((size,5))
    for i in range(len(image)):
        batch[i] = h5py.File(source+f'/fMRI_train/{image[i]}.mat', 'r')['SM_feature'][()]
        y_train[i] = y[image[i]]
    
    return batch, y_train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, BatchNormalization, PReLU, Reshape , MaxPooling3D,MaxPool3D
import tensorflow.keras.backend as K
def weighted_NAE(yTrue,yPred):
    weights = K.constant([.3, .175, .175, .175, .175])

    return K.sum(weights*K.sum(K.abs(yTrue-yPred))/K.sum(yPred))
model = Sequential()
model.add(Conv3D(200, (3,3,3), input_shape = (53, 52, 63, 53), activation='relu', data_format='channels_first'))
model.add(MaxPool3D((2,2,2)))
model.add(Dropout(0.2))
model.add(Conv3D(200, (3,3,3), activation='relu', data_format='channels_first'))
model.add(MaxPool3D((2,2,2)))
model.add(BatchNormalization()) 

model.add(Conv3D(200, (3,3,3), activation ='relu', data_format='channels_first'))
model.add(MaxPool3D((2,2,2)))
model.add(Dropout(0.2))
model.add(Conv3D(200, (3,3,3), activation='relu', data_format='channels_first'))
model.add(MaxPool3D((2,2,2)))

model.add(Flatten())
model.add(BatchNormalization()) 
model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='relu'))
model.compile(loss='mse',
              optimizer='adam',
              metrics='mse')
batch_size = 10
nb_batchs = 10

for e in range(150):
    print(f'epoch {e}/150')
    loss = 0.
    acc = 0.
    for batch in range(nb_batchs):
        train_x, train_y = get_batch(batch, batch_size)
        loss_batch, acc_batch = model.train_on_batch(train_x, train_y)
        loss += loss_batch
        acc += acc_batch
    print(f'Loss: {loss / nb_batchs}')
    print(f'Acc: {acc / nb_batchs}')
model.save('model.h5')
img_test = []
for name in os.listdir(source+'/fMRI_test'):
    img_test.append(int(name[:5]))
img_test.sort()
t = h5py.File(source+f'/fMRI_test/{img_test[0]}.mat', 'r')['SM_feature'][()]
model.predict(np.array([t])).flatten()
pred_model = np.array([])
for img in tqdm(img_test):
    t = h5py.File(source+f'/fMRI_test/{img}.mat', 'r')['SM_feature'][()]
    if len(pred_model) == 0:
        pred_model = model.predict(np.array([t])).flatten()
    else:
        pred_model = np.append(pred_model, model.predict(np.array([t])).flatten())
sample_submission = pd.read_csv("/kaggle/input/trends-assessment-prediction/sample_submission.csv")
pred=pd.DataFrame()
pred["Id"]=sample_submission.Id
pred["Predicted"]=pred_model
pred.to_csv('out.csv', index=False)



