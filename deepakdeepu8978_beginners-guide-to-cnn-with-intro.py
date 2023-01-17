import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

import tensorflow.keras as keras

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
#now as label is the target we will saparate for training 

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

X_train.shape
X_train.isnull().sum()

Y_train.isnull().sum()
sns.countplot(Y_train)
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
plt.figure( figsize = (10,10))

for i in range(1,10):

    plt.subplot(3,4,i)

    plt.imshow(X_train[i+1].reshape([28,28]),cmap="gray")

Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
model = keras.models.Sequential([

    keras.layers.Conv2D(128, (3,3), input_shape=(28,28,1), activation='relu'),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(256, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(512, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),

    keras.layers.Dense(128),

    keras.layers.BatchNormalization(),

    keras.layers.Activation('relu'), 

    keras.layers.Dropout(0.1),

    keras.layers.Dense(64),

    keras.layers.BatchNormalization(),

    keras.layers.Activation('relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer = 'adam',loss = "categorical_crossentropy", metrics=["accuracy"])

model.summary()
X_train ,X_test_test,Y_train,Y_test= train_test_split(X_train,Y_train,test_size=0.2)
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=5,baseline=0.99)



model.fit(X_train,Y_train, epochs=20, callbacks=[es], validation_data = (X_test_test,Y_test))
model.evaluate(X_test_test,Y_test)


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
checkpoint = ModelCheckpoint('best_weigths.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)



history = model.fit_generator(datagen.flow(X_train,Y_train),

                              epochs = 30, validation_data = (X_test_test,Y_test),

                              verbose = 1, callbacks=[es,checkpoint])
X_test_np = np.array(test)



print(X_test_np.shape)



plt.figure( figsize = (10,10))

for i in range(1,20):

    toBePredicted = X_test_np[i].reshape(-1,28,28,1)

    result2 = model.predict(toBePredicted)

    print(np.argmax(result2,axis = 1))

    

    plt.subplot(10,4,i)

    plt.imshow(X_test_np[i].reshape([28,28]))
# predict results

results = model.predict(X_test_np)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

print(results)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
acc = history.history['acc']

epochs_s = range(0,30)



plt.plot(epochs_s, acc , label='accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')



plt.title('accuracy vs epochs')

plt.legend()