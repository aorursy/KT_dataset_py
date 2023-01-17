# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras import Model
from keras import layers as L
dict_size=100 #numero de palavras no dicionario

inp  = L.Input((10,))
x = inp

x = L.Embedding(dict_size, 64) (x)
x = L.Bidirectional(L.CuDNNLSTM(35)) (x)
x = L.Dropout(0.1)(x)
x = L.Dense(16, activation='relu') (x)

for k in range(3):
    xbkp = L.Dense(16, activation='relu') (x)
    x = L.Dense(16, activation='relu') (xbkp)
    x = L.Dense(16, activation='relu') (xbkp)
    x = L.Add()([x, xbkp])
    
x = L.Dense(1,activation='sigmoid') (x)

out = x
model = Model(inputs=inp, outputs=out)
nchannels = 12
n_categs = 10

inp = L.Input( (28,28,3) )

inp2 = L.Input( (28,28,3) )

x = inp
for k in range(3):
    x = L.Conv2D(nchannels, (3,3), activation='relu', padding = 'same') (x)
    x = L.MaxPool2D( (2,2) ) (x)
    nchannels*=2

x = L.Flatten() (x)
x = L.Dense(n_categs, activation='softmax') (x)

out = x
model = Model(inputs=[inp, inp2], outputs = out)
model.summary(line_length=80)
for l in model.layers:
    print(l.trainable)
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)
n=0
for ll in model.layers:
    if n<10:
        ll.trainable=False
    n+=1
    print(ll.trainable)

entrada=np.linspace(0,9,10).reshape(1,-1)
print(entrada)
entrada.shape
model.predict(entrada)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
import matplotlib.pyplot as plt
x_train.shape
plt.imshow(x_train[0])
from keras import Model
from keras import layers as L
from keras import backend as K
inp=L.Input( (28,28) )

x=inp
x=L.Lambda(lambda z: K.expand_dims(z,3))(x)

nFilters=10
depths = 5
for p in range(3):
    for k in range(depths):
        x = L.SeparableConv2D(nFilters, (3,3), activation='relu', padding='same')(x)
        x = L.BatchNormalization()(x)
        
    if p<2:
        x = L.MaxPooling2D((2,2))(x)    
    nFilters*=2
    
x = L.Conv2D(10,(7,7), activation='softmax')(x)
x = L.Lambda(lambda z: K.squeeze(K.squeeze(z,axis=1),axis=1) )(x)
    
out = x

model = Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
model.summary(line_length=80)
y_train.reshape((-1,1)).shape
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
def step_decay(epoch):
    initial_lrate = 1e-3               
    drop = 0.6
    epochs_drop = 15
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    
    if (lrate < 1e-6):
        lrate = 1e-6
      
    print('Changing learning rate to {}'.format(lrate))
    return lrate
lrate = LearningRateScheduler(step_decay)
earlystopper = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint('model-fashion.h5', verbose=1, save_best_only=True)

#model.fit(x_train, y_train.reshape((-1,1)), validation_split=0.15, epochs=50, batch_size=64,
#          callbacks=[lrate,earlystopper, checkpointer])
model.load_weights('model-fashion.h5')
model.evaluate(x_train, y_train.reshape(-1,1))
model.evaluate(x_test, y_test.reshape(-1,1))
