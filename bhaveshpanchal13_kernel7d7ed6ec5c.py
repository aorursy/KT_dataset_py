# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from tensorflow.keras.optimizers import RMSprop,Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau
train=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
test.head(3)

test=test.drop('id',axis=1)
X_train=train.drop('label',axis=1)

Y_train=train.label
X_train=X_train/255

test=test/255
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
Y_train=to_categorical(Y_train)
X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.15)
datagen = ImageDataGenerator(

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





datagen.fit(X_train)
model = Sequential()



model.add(Conv2D(filters = 12, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 12, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

#model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 24, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 24, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

#model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

#model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu", kernel_initializer='he_normal'))

model.add(Dropout(0.37))

model.add(Dense(10, activation = "softmax"))
model.summary()
optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5,

                                            min_lr=0.00001)
epochs=5 #change this to 30 if you need to get better score

batch_size=128
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
fig,ax=plt.subplots(2,1)

fig.set

x=range(1,1+epochs)

ax[0].plot(x,history.history['loss'],color='red')

ax[0].plot(x,history.history['val_loss'],color='blue')



ax[1].plot(x,history.history['accuracy'],color='red')

ax[1].plot(x,history.history['val_accuracy'],color='blue')

ax[0].legend(['trainng loss','validation loss'])

ax[1].legend(['trainng acc','validation acc'])

plt.xlabel('Number of epochs')

plt.ylabel('accuracy')
test=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
test_id=test.id

test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)
y_pre=model.predict(test)   

y_pre=np.argmax(y_pre,axis=1)
sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)