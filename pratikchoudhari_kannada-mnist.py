import pandas as pd



train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

test=test.drop('id',axis=1)
from sklearn.model_selection import train_test_split



x = train.drop('label',axis=1)

y = train.label

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.15, random_state=42,shuffle=False)

from keras.utils import normalize

from keras.utils.np_utils import to_categorical



x_train=x_train/255

x_valid=x_valid/255

test=test/255

x_train=x_train.values.reshape(-1,28,28,1)

x_valid=x_valid.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
y_train
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

for i in range(21):

  plt.subplot(3,7,i+1)

  plt.imshow(x_train[i][:,:,0], cmap='binary')

  plt.title(y_train[i])
y_train = to_categorical(y_train)

y_valid = to_categorical(y_valid)
from keras.preprocessing.image import ImageDataGenerator

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

        horizontal_flip=0.1,  # randomly flip images

        vertical_flip=0.1)  # randomly flip images

datagen.fit(x_train)
x_valid[2][15]
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPool2D, Dropout

from keras.callbacks import ReduceLROnPlateau

import warnings

warnings.filterwarnings('ignore')



learning_rate_reduction = ReduceLROnPlateau(monitor='acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001)

epochs = 10

model = Sequential()

model.add(Conv2D(filters = 120, kernel_size = (5,5),activation ='relu', input_shape = (28,28,1), padding='Same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 120, kernel_size = (5,5),activation ='relu', padding='Same'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(100, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=epochs,callbacks=[learning_rate_reduction],validation_data=(x_valid, y_valid),batch_size=200)
plt.figure(figsize=(15,8))

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(20,5))

x=range(1,1+epochs)

ax1.plot(x,history.history['loss'],color='red')

ax1.plot(x,history.history['val_loss'],color='blue')

ax1.legend(['trainng loss','validation loss'])

ax2.plot(x,history.history['accuracy'],color='red')

ax2.plot(x,history.history['val_accuracy'],color='blue')

ax2.legend(['trainng acc','validation acc'])

ax1.set_xlabel('Number of epochs')

ax1.set_ylabel('accuracy')

ax2.set_xlabel('Number of epochs')

ax2.set_ylabel('loss')
import numpy as np

predicted = model.predict(test)

submit=np.argmax(predicted,axis=1)
submit = pd.DataFrame(submit)

submit['id'] = submit.index

submit.columns = ['label', 'id']

submit = submit[['id','label']]
submit.to_csv('submission.csv',index=False)