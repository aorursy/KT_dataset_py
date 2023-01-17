# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras import layers

from keras import models

from keras import optimizers

from keras import backend as k

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping



from IPython.display import Image





import matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import random

import sys

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

ds_ = pd.read_csv("../input/fer2018/fer20131.csv")

ferSubmission = pd.read_csv("../input/fer2018/ferSubmission.csv")
#emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,'Sad': 4, 'Surprise': 5, 'Neutral': 6}



# replace literal labels for numeric

classes = {0:'Angry', 1:'Disgust', 2: 'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

ds = ds_.applymap(lambda s: classes.get(s) if s in classes else s)

ds['emotion'].value_counts().plot.bar()

print ("\n" + "-> Classes: " + str(ds['emotion'].unique()))
ds.head()
ds_.head()
def get_in_out_data(dataset):

    x = list(dataset["pixels"])

    X = []

    

    for i in range(len(x)):

        each_pixel = [int(num) for num in x[i].split()]

        X.append(each_pixel)

        

    X = np.array(X)

    X = X.reshape(X.shape[0], 48, 48,1)  # reshape to a 48x48 grayscale (1 pixel) format

    X = X.astype("float32")

    X /= 255 # normalize

    

    Y = dataset.emotion.values

    Y = np_utils.to_categorical(Y)

    

    return X,Y
# training set

train_ds = ds_[ds_.Usage == 'Training']

train_ds = train_ds.reset_index(drop=True)

x_train, y_train = get_in_out_data(train_ds)



# validation set

test_ds = ds_[ds_.Usage == 'PrivateTest']

test_ds = test_ds.reset_index(drop=True)

x_test, y_test = get_in_out_data(test_ds)



print("x_train: " + str(x_train.shape))

print("y_train: " + str(y_train.shape))

print("x_test:  " + str(x_test.shape))

print("y_test:  " + str(y_test.shape))
emotion_classes = ['Angry','Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']



fig = plt.figure(figsize=(20,20))

for i in range(0, 100):

    input_img = x_train[i:(i+1),:,:,:]

    ax = fig.add_subplot(10,10,i+1)

    ax.imshow(input_img[0,:,:,0], cmap=plt.cm.gray)

    plt.title("Emotion: {0}".format(emotion_classes[train_ds.emotion.values[i]]))

    plt.xticks(np.array([]))

    plt.yticks(np.array([]))

    plt.tight_layout()

plt.show()
# Adaptive Network model

adaptNet_model = models.Sequential()

adaptNet_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))

adaptNet_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))

adaptNet_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

adaptNet_model.add(layers.Dropout(0.05))



adaptNet_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

adaptNet_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

adaptNet_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

adaptNet_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

adaptNet_model.add(layers.Dropout(0.12))



adaptNet_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))

adaptNet_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))

adaptNet_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))

adaptNet_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

adaptNet_model.add(layers.Dropout(0.12))



adaptNet_model.add(layers.Flatten())  

adaptNet_model.add(layers.Dense(128, activation='relu'))

adaptNet_model.add(layers.Dropout(0.25))

adaptNet_model.add(layers.Dense(128, activation='relu'))

adaptNet_model.add(layers.Dropout(0.25))

adaptNet_model.add(layers.Dense(7, activation='softmax'))



adaptNet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



adaptNet_model.summary()
# callback definitions

earlystop = EarlyStopping(patience=5)

callbacks = [earlystop]
epochs = 32

batch_size = 128



adptNet_model = adaptNet_model.fit(x_train,

                                y_train, 

                                callbacks=callbacks, 

                                epochs=epochs, 

                                batch_size=batch_size,

                                validation_data=(x_test, y_test), shuffle=True, verbose=1)



# save model

adaptNet_model.save('/kaggle/working/adaptNet_model_v1.h5')
def plotModelFitResults(model):

    acc = model.history['accuracy']

    val_acc = model.history['val_accuracy']

    loss = model.history['loss']

    val_loss = model.history['val_loss']



    epochs = range(len(acc))



    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



    plt.show()
plotModelFitResults(adptNet_model)
def evaluateModel(model):

    # evaluate model

    score = model.evaluate(x_test, y_test, verbose=0)

    print ("model %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

evaluateModel(adaptNet_model)
# prediction set

#ds_.Usage.value_counts()

prediction_ds = ds_[ds_.Usage == 'PublicTest']

prediction_ds.head(10)
prediction_data = prediction_ds.iloc[[44,257,43,20,27,31,36,17,666],:]

# replace literal labels for numeric

classes = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,'Sad': 4, 'Surprise': 5, 'Neutral': 6}



prediction_ds = prediction_ds.applymap(lambda s: classes.get(s) if s in classes else s)



prediction_data = prediction_data.reset_index(drop=True)

x_prediction, y_prediction = get_in_out_data(prediction_data)



print("x_prediction: " + str(x_prediction.shape))



print("y_prediction: " + str(y_prediction.shape))
# predict emotion using loaded model

predicted_data = adaptNet_model.predict(x_prediction)

emotion_classes = ['Angry','Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']



fig = plt.figure(figsize=(6,8))

for i in range(0, 9):

    input_img = x_prediction[i:9,:,:,:]

    ax = fig.add_subplot(3,3,i+1)

    ax.imshow(input_img[0,:,:,0], cmap=plt.cm.gray)

    indx = np.where(predicted_data[i] == np.amax(predicted_data[i]))

    idx = int(indx[0])

    plt.title("Emotion: {0}".format(emotion_classes[idx]))



plt.show()
adaptNet_model = models.load_model('/kaggle/working/adaptNet_model_v1.h5')

adaptNet_model.summary()