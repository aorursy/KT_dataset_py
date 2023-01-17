import h5py

import matplotlib.pyplot as plt

from tqdm import tqdm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nilearn as nl

import nilearn.plotting as nlplt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, BatchNormalization, PReLU, Reshape , MaxPooling3D,MaxPool3D

import tensorflow.keras.backend as K

from sklearn.impute import KNNImputer

import tensorflow as tf

from keras.callbacks import ModelCheckpoint

from sklearn.decomposition import PCA, TruncatedSVD

import gc

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from itertools import combinations, product

import random

from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")
np.random.seed = 42



ids=np.array(train.Id)

X_train_id, X_pretest_id, Y_train, Y_pretest = train_test_split(ids, train.drop(["Id"], axis=1), test_size=0.1)

impute = KNNImputer(n_neighbors=40)

Y_train_knn = impute.fit_transform(Y_train)

Y_pretest_knn = impute.transform(Y_pretest)
def weighted_NAE(yTrue,yPred):

    weights = K.constant([.3, .175, .175, .175, .175], dtype=tf.float32)

    return K.sum(weights*K.sum(K.abs(yTrue-yPred))/K.sum(yPred))
lr=0.001

activ='relu'



strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    Model = Sequential()

    Model.add(Conv3D(100, (3,3,3), input_shape = (53, 52, 63, 53), activation =activ))

    Model.add(Conv3D(100, (3,3,3), activation =activ))

    Model.add(MaxPool3D((2,2,2)))



    

    Model.add(Conv3D(200, (3,3,3), activation =activ))

    Model.add(Conv3D(200, (3,3,3), activation =activ))

    Model.add(MaxPool3D((2,2,2)))



    

    Model.add(Conv3D(500, (2,2,2), activation =activ))

    Model.add(MaxPool3D((2,2,2)))

    Model.add(BatchNormalization())    

    

    Model.add(Flatten())

    Model.add(Dense(5000, activation=activ))

    Model.add(Dropout(0.1))



    Model.add(Dense(2000, activation=activ))

    Model.add(Dropout(0.1))



    Model.add(Dense(500, activation=activ))

    Model.add(Dropout(0.1))



    Model.add(Dense(5, activation = activ))



    Model.compile(loss='mse' , optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[weighted_NAE])

    print(Model.summary())

with h5py.File('foo3.hdf5','w') as myfile:

    for i ,name in enumerate(X_train_id):

        file=str(name) + ".mat"

        link = "/kaggle/input/trends-assessment-prediction/fMRI_train/" 



        myfile[str(i)] = h5py.ExternalLink(link+file, '/SM_feature')
val_loss =[]

loss=[]

NAE=[]

val_NAE=[]
batch_size=10

epochs=10





batch = np.zeros((batch_size,53, 52, 63, 53))

for e in range(epochs):

    with h5py.File('foo3.hdf5','r') as myfile:        



        batch_n = random.sample(list(np.arange(len(X_train_id))), batch_size)



        for c, i  in enumerate(batch_n):

            batch[c] = myfile[str(i)]



        hist = Model.fit(batch, Y_train_knn[batch_n], validation_split=0.1)

        loss+=hist.history["loss"]

        val_loss+=hist.history["val_loss"]

        NAE+=hist.history["weighted_NAE"]

        val_NAE+=hist.history["val_weighted_NAE"]

        
plt.ylim(0,10)

plt.plot(np.arange(len(val_NAE)),val_NAE, c='r')

plt.plot(np.arange(len(NAE)),NAE, c='b')

plt.show()
val_loss