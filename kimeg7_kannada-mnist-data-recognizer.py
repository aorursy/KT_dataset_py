# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import visualization tools

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow



# Import sklearn API

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Import Keras API

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau

root = "../input/Kannada-MNIST"

train = pd.read_csv(os.path.join(root,'train.csv'))

test = pd.read_csv(os.path.join(root,'test.csv'))

final = pd.read_csv(os.path.join(root,'sample_submission.csv'))
test = test.drop('id',axis=1)
X_train=train.drop('label',axis=1)

Y_train=train.label
X_train=X_train/255

test=test/255
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
Y_train=to_categorical(Y_train)
X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.2)
data_tweaker = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





data_tweaker.fit(X_train)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs=30

batch_size=64
train_info = model.fit_generator(data_tweaker.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
fig,ax=plt.subplots(2,1)

fig.set

x=range(1,1+epochs)

ax[0].plot(x,train_info.history['loss'],color='red')

ax[0].plot(x,train_info.history['val_loss'],color='blue')



ax[1].plot(x,train_info.history['accuracy'],color='red')

ax[1].plot(x,train_info.history['val_accuracy'],color='blue')

ax[0].legend(['Training Loss','Validation Loss'])

ax[1].legend(['Training Acc','Validation Acc'])

plt.xlabel('Number of Epochs')

plt.ylabel('Accuracy')
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)



y_real = np.argmax(y_test,axis=1)



conf_mat = confusion_matrix(y_real, y_pred)

conf_mat = pd.DataFrame(conf_mat, index=range(0,10), columns=range(0,10))



plt.figure(figsize=(8,6))

sns.set(font_scale=1.4)#for label size

sns.heatmap(conf_mat, annot=True,annot_kws={"size": 16},cmap=plt.cm.Blues)# font size
test_data = pd.read_csv(os.path.join(root,'test.csv'))



#Drop the ID column (or else what's the purpose of doing all these?)

test_data = test_data.drop('id',axis=1)



#Scale values to [0..1]

test_data = test_data/255



#Reshape test data for keras model to be used for test data prediction

test_data = test_data.values.reshape(-1,28,28,1)
pred = model.predict(test_data)     

final['label'] = np.argmax(pred, axis=1) 

final['id'] = np.arange(len(final['label']))

final.to_csv('../working/submission.csv',index=False)