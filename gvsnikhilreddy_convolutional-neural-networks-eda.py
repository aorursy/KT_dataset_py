import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

from keras.layers import Dropout, Flatten ,BatchNormalization , MaxPool2D

from keras.layers.convolutional import Conv2D

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import os
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")
train.columns
train['label'][1:10]
train.shape
x=train.drop(['label'],axis=1)

y1=y=train['label']

y = to_categorical(y)
x.head()
sns.countplot(y1)
x=x/255.0

test=test/255.0

x = np.array(x).reshape(-1,28,28,1)

test = np.array(test).reshape(-1,28,28,1)

x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=4)

x_train.shape,y_train.shape,x_test.shape,y_test.shape
fig, axis = plt.subplots(10, 10, figsize=(20, 20))

for i, ax in enumerate(axis.flat):

    ax.imshow(x_train[i].reshape(28,28))

    ax.axis('off')

    ax.set(title = f"Real Number is {y_train[i].argmax()}")
model = Sequential()

#First

model.add(Conv2D(filters = 64, kernel_size = (3,3) ,activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 56, kernel_size = (3,3),activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.2))



#Second

model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))

model.add(Conv2D(filters = 48, kernel_size = (3,3),activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.2))



#Third

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(128, activation = "relu"))

model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.4))



#Output

model.add(Dense(10, activation = "softmax"))





model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

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

        vertical_flip=False)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=56),

                              epochs = 10, validation_data = (x_test,y_test),

                              verbose = 2, steps_per_epoch=660)

plt.figure()

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))

ax1.plot(history.history['loss'])

ax1.plot(history.history['val_loss'])

ax1.legend(['training','validation'])

ax1.set_title('loss')

ax1.set_xlabel('epoch')



ax2.plot(history.history['accuracy'])

ax2.plot(history.history['val_accuracy'])

ax2.legend(['training','validation'])

ax2.set_title('Acurracy')

ax2.set_xlabel('epoch')







score =model.evaluate(x_test,y_test,verbose=0)

print('Test Score:',score[0])

print('Test Accuracy:',score[1])
y_pred = model.predict(x_test)

X_test__ = x_test.reshape(x_test.shape[0], 28, 28)



fig, axis = plt.subplots(5, 5, figsize=(20, 20))

for i, ax in enumerate(axis.flat):

    ax.imshow(X_test__[i])

    ax.axis('off')

    ax.set(title = f"Real Number is {y_test[i].argmax()}\nPredict Number is {y_pred[i].argmax()}");

results = model.predict(test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("CNN_Digit_Recognizer.csv",index=False)