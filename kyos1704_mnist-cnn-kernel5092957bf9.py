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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train)
print(test)
Y_train = train["label"]

X_train = train.drop(labels=["label"],axis=1)



del train



g = sns.countplot(Y_train)

Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalize

X_train = X_train/255.0

test = test/255.0
# Reshape



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
g = plt.imshow(X_train[0][:,:,0])
#define model

from keras.models import Model

from keras.engine.topology import Input

from keras.layers.core import Flatten, Dense,Reshape,Dropout,Activation

from keras.layers.convolutional import Conv2D,UpSampling2D

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.layers.normalization import BatchNormalization

import numpy as np

import tensorflow as tf

from keras import backend as K





def GetModel():

    input = Input((28,28,1))

    layer1 = Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1))(input)

    layer2 = Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu')(layer1)

    layer3 = MaxPooling2D(pool_size=(2,2))(layer2)

    layer4 = Dropout(0.25)(layer3)

    

    

    out_layer1 = Flatten()(layer4)

    out_layer2 = Dense(256,activation='relu')(out_layer1)

    out_layer3 = Dropout(0.5)(out_layer2)

    out_layer4 = Dense(10,activation='softmax')(out_layer3)

    model = Model(inputs=[input],outputs=[out_layer4])

    return model



GetModel().summary()
optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
model = GetModel()

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs = 4

batch_size = 86
# Fit the model



#history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

#                              epochs = epochs, validation_data = (X_val,Y_val),

#                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

#                              , callbacks=[learning_rate_reduction])



history = model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val,Y_val))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)