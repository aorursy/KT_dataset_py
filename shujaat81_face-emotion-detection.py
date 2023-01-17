# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.utils import to_categorical



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013/fer2013.csv')
data.head(5)
data.Usage.value_counts()
data_train = data[data['Usage']== 'Training'].copy()

data_val = data[data['Usage']=='PublicTest'].copy()

data_test = data[data['Usage']=='PrivateTest'].copy()
print('train shape: {}, \nvalidation shape: {}, \ntest shape: {}'.format(data_train.shape,data_val.shape,data_test.shape))
#Initialize Parameters

num_classes = 7

epochs = 55

batch_size = 64

num_features = 32

width, height = 48,48
#Perform CRNO (CRNO stands for Convert,Reshape, Normalize, one-hot-encoding)

def CRNO(df,dataName):

    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])

    data_X = np.array(df['pixels'].tolist(),dtype='float32').reshape(-1,width,height,1)/255.0

    data_Y = to_categorical(df['emotion'],num_classes)

    print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))

    return data_X,data_Y
train_X,train_Y = CRNO(data_train,'train')

val_X,val_Y = CRNO(data_val,'validation')

test_X,test_Y = CRNO(data_test,'test')
#Import Libraries before model creation

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,BatchNormalization

from keras.layers import Dense,Dropout,Activation,Flatten

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix,classification_report

import matplotlib.pyplot as plt

import seaborn as sns

#Build the model

model = Sequential()

#Module1 conv<<conv<<batchnorm<<relu<<maxpooling<<dropout

model.add(Conv2D(2*num_features,kernel_size=(3,3),padding='same',data_format='channels_last',input_shape=(width, height, 1)))

model.add(Conv2D(2*num_features,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D())

model.add(Dropout(rate=0.2))

#Module2 conv<<conv<<batchnorm<<relu<<maxpool<<dropout

model.add(Conv2D(2*2*num_features,kernel_size=(3,3),padding='same'))

model.add(Conv2D(2*2*num_features,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D())

model.add(Dropout(rate=0.2))

#Module3 conv<<conv<<batchnorm<<relu<<maxpool<<dropout

model.add(Conv2D(2*2*2*num_features,kernel_size=(1,1),padding='same'))

model.add(Conv2D(2*2*2*num_features,kernel_size=(1,1),strides=(2,2)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D())

model.add(Dropout(rate=0.2))

#Module4 fc<<batchnorm<<fc<<batchnorm<<dropout<<softmax

model.add(Flatten())

model.add(Dense(units=128))

model.add(BatchNormalization())

model.add(Dense(units=128))

model.add(BatchNormalization())

model.add(Dropout(rate=0.2))

model.add(Dense(num_classes,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999),metrics=['accuracy'])

model.summary()







es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)



history = model.fit(train_X,train_Y,batch_size=batch_size,epochs=50,verbose=2,callbacks=[es],validation_split=0,validation_data=(val_X,val_Y),shuffle=True)
test_true = np.argmax(test_Y, axis=1)

test_pred = np.argmax(model.predict(test_X), axis=1)

print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))