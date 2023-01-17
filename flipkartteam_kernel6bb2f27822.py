import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import skimage.transform

import os
X=np.ndarray(shape=(4000,299,299,3),dtype='float32')

Y=np.ndarray(shape=(4000,4),dtype='float32')

directory='../input/octkermany4000image/train5k/train5k/CNV'

i=0

for file in sorted(os.listdir(directory)):

    image=plt.imread(directory+'//'+file)

    newimage=skimage.transform.resize(image, (299, 299,3), mode='constant')

    X[i]=newimage

    Y[i]=np.array([1,0,0,0])

    i=i+1

    

directory='../input/octkermany4000image/train5k/train5k/DME'

for file in sorted(os.listdir(directory)):

    image=plt.imread(directory+'//'+file)

    newimage=skimage.transform.resize(image, (299, 299,3), mode='constant')

    X[i]=newimage

    Y[i]=np.array([0,1,0,0])

    i=i+1

    

directory='../input/octkermany4000image/train5k/train5k/DRUSEN'

for file in sorted(os.listdir(directory)):

    image=plt.imread(directory+'//'+file)

    newimage=skimage.transform.resize(image, (299, 299,3), mode='constant')

    X[i]=newimage

    Y[i]=np.array([0,0,1,0])

    i=i+1  

 

directory='../input/octkermany4000image/train5k/train5k/NORMAL'

for file in sorted(os.listdir(directory)):

    image=plt.imread(directory+'//'+file)

    newimage=skimage.transform.resize(image, (299, 299,3), mode='constant')

    X[i]=newimage

    Y[i]=np.array([0,0,0,1])

    i=i+1

    

print('done')    
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain,Ytest = train_test_split(X,Y, test_size=0.2) # Re-comm
import keras

from keras.applications.xception import Xception

from keras.layers import Activation, Dense

from keras.models import Model
del X,Y


model=Xception(include_top=True,input_shape=(299,299,3), weights='../input/xception/xception_weights_tf_dim_ordering_tf_kernels.h5')

model.layers.pop()

x = Dense(4,activation='softmax')(model.layers[-1].output)

newmodel=Model(input = model.input, output = [x])

print(newmodel.summary())
import keras

batch_size= 16

epochs = 50



newmodel.compile(loss=keras.losses.categorical_crossentropy ,optimizer=keras.optimizers.Adam() ,metrics=['accuracy'])
trained_model=newmodel.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=epochs,shuffle=True)
newmodel.save('oct.h5')