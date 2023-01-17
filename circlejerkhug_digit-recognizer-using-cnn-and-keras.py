# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt # to plot charts

from sklearn.model_selection import train_test_split

from PIL import Image

from scipy import ndimage



import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate

from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.regularizers import l2



np.random.seed(7)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the data from the input CSV files. Use relative path.

train_path = '..//input//train.csv'

test_path = '..//input//test.csv'



train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)



train_df.info()

test_df.info()
'''

Training data has column "label" which is the Y value. So, assign all columns apart from "label" to X_train and assign the

"label" column value to Y_train

'''

X_train=train_df.drop("label",axis=1).values

Y_train=train_df["label"].values



print ('Shape of X_train >>',X_train.shape)

print ('Shape of Y_train >>',Y_train.shape)



'''

Test data are not labeled. So, assigning all to X_test

'''

X_test=test_df.values



print ('Shape of X_test >>',X_test.shape)
#prepare data e.g. one hot conversion

num_classes = len(np.unique(Y_train))

num_classes



X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

Y_train = keras.utils.to_categorical(Y_train, num_classes)



print(X_train.shape, 'train samples')

print(X_test.shape, 'test samples')

print(Y_train.shape, 'target train samples')
# convert int to float. This helps avoid rounding of parameter values

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



# normalise

X_train /= 255

X_test /= 255
# Visualize a sample of the training data



index  = 120

k = X_train[index,:]

k = k.reshape((28, 28))

plt.title('Label is {label}'.format(label= np.argmax(Y_train[index])))

plt.imshow(k, cmap='gray')
# Inception block

def inception_block(inputs):

    tower_one = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)

    tower_one = Conv2D(6, (1,1), activation='relu', border_mode='same')(tower_one)



    tower_two = Conv2D(6, (1,1), activation='relu', border_mode='same')(inputs)

    tower_two = Conv2D(6, (3,3), activation='relu', border_mode='same')(tower_two)



    tower_three = Conv2D(6, (1,1), activation='relu', border_mode='same')(inputs)

    tower_three = Conv2D(6, (5,5), activation='relu', border_mode='same')(tower_three)

    x = concatenate([tower_one, tower_two, tower_three], axis=3)

    return x



# Creating model with the inception block

def inception_model(x_train):



    inputs = Input(x_train.shape[1:])



    x = inception_block(inputs)

        

    x = Dropout(0.25)(x)



    x = Conv2D(32, (3, 3), padding='same')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3), padding='same')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.25)(x)

    

    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)

    x = Dropout(0.5)(x)

    predictions = Dense(10, activation='softmax')(x)



    model = Model(input=inputs, output=predictions)



    model.compile(loss='categorical_crossentropy',

                 optimizer='adam',

                 metrics=['accuracy'])

    return model
model = inception_model(X_train)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train,

              batch_size=100,

              epochs=100,

              validation_split=0.1,

              shuffle=True)
scores_train = model.evaluate(X_train, Y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis = 1)

predictions
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
result=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

result.to_csv("mnist_cnn_only_v1.csv", index=False, header=True)