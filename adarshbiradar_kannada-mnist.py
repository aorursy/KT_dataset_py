#importing librabries

import pandas as pd

import numpy as np

import keras

from keras.models import Sequential

from keras.layers.core import Dropout,Activation,Dense,Flatten

from keras.layers.convolutional import Convolution2D,MaxPooling2D

from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
data=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')
data.head()
#reshaping

data.iloc[3,1:].values.reshape(28,28)
X=data.iloc[:,1:].values.reshape(len(data),28,28,1)

test_1=test.iloc[:,1:].values.reshape(len(test),28,28,1)

y=data.iloc[:,0].values
X[1]
y[1]
df_y=keras.utils.to_categorical(y,num_classes=10)
X=np.array(X)

test_2=np.array(test_1)

df_y=np.array(df_y)

X=X/255

test_3=test_2/255
y
df_y
X.shape
x_train, x_test, y_train, y_test = train_test_split(X,df_y,test_size=0.2,random_state=44)
model=Sequential()

model.add(Convolution2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model.add(Convolution2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Convolution2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))

model.add(Convolution2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 20

batch_size = 36
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





datagen.fit(X)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
results = model.predict(test_3)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

id_ = np.arange(0,results.shape[0])
save = pd.DataFrame({'id':id_,'label':results})

print(save.head(10))

save.to_csv('submission.csv',index=False)
save
