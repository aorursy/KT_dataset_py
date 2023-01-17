import os

import zipfile

import cv2

import numpy as np

import pandas as pd

from keras.optimizers import *

from keras.models import Sequential

from keras.layers import *



  

def base_dir11(path):

    imgs = {}

    for f in os.listdir(path):

        fname = os.path.join(path, f)

        #print(fname)

        imgs[f] = cv2.imread(fname)

    return imgs



base_dir = base_dir11 ("../input/train/train")

base_dir1 = base_dir11 ("../input/test/test")
type(base_dir)



cnn_classifier = Sequential()

cnn_classifier.add(Conv2D(64, (3, 3), input_shape = (32, 32, 3),padding='same'))

cnn_classifier.add(BatchNormalization())

cnn_classifier.add(Activation('elu'))     

cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))



# 2nd conv. layer

cnn_classifier.add(Conv2D(128, (3, 3),padding='same'))

cnn_classifier.add(BatchNormalization())

cnn_classifier.add(Activation('elu'))     

cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))



# 3nd conv. layer

cnn_classifier.add(Conv2D(256, (3, 3),padding='same'))

cnn_classifier.add(BatchNormalization())

cnn_classifier.add(Activation('elu'))     

cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))





# 4nd conv. layer

cnn_classifier.add(Conv2D(512, (3, 3),padding='same'))

cnn_classifier.add(BatchNormalization())

cnn_classifier.add(Activation('elu'))     

cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))





cnn_classifier.add(Flatten())

    

cnn_classifier.add(Dropout(0.4))

cnn_classifier.add(Dense(64, activation='elu'))

cnn_classifier.add(Dense(1, activation='sigmoid'))

cnn_classifier.compile(optimizer = 'adam', 

                       loss = 'binary_crossentropy', 

                       metrics = ['accuracy'])





df_csv = pd.read_csv('../input/train.csv')



X_train = []

Y_train = []



for index,row in df_csv.iterrows():

    X_train.append(base_dir[row['id']])

    Y_train.append(int(row['has_cactus']))



X_train = np.array(X_train)

Y_train = np.array(Y_train)



X_test = np.array([base_dir1[f] for f in base_dir1])



cnn_classifier.fit(X_train, Y_train, epochs=20, batch_size=10)



pred = cnn_classifier.predict(X_test)





df = pd.DataFrame({

    'id': [f for f in base_dir1],

    'has_cactus': [int(x[0] >= 0.5) for x in pred]

})



X_train.shape

Y_train.shape
df.to_csv('CNN.csv', index=False)