# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.layers import *

from keras.models import *

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'svg'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/fashion-mnist_train.csv")

train_df.head(5)
batch_size = 128

num_classes = 10

epochs = 10

img_rows, img_cols = 28, 28



y_train = keras.utils.to_categorical(train_df.label.values, num_classes)

print("y_train: ", y_train.shape)



x_train = np.array([row.reshape((img_rows, img_cols, 1)) 

           for row in train_df.drop('label', axis=1, inplace=False).values])

print("x_train: ", x_train.shape)
def model(input_shape):

    x_input = Input(input_shape)

    

    x = Conv2D(20, (5, 5), strides = (1, 1), name = 'conv0')(x_input)

    x = BatchNormalization(axis = 3, name = 'bn0')(x)

    x = Activation('relu')(x)

    

    x = MaxPooling2D((2, 2), strides = (2,2), name='max_pool0')(x)

    

    x = Conv2D(25, (3, 3), strides = (1, 1), padding="same", name = 'conv1')(x)

    x = BatchNormalization(axis = 3, name = 'bn1')(x)

    x = Activation('relu')(x)

    

    x = MaxPooling2D((3, 3), strides = (2,2), name='max_pool1')(x)

    

    x = Conv2D(30, (1, 1), strides = (1, 1), padding="same", name = 'conv2')(x)

    x = BatchNormalization(axis = 3, name = 'bn2')(x)

    x = Activation('relu')(x)

    

    x = Dropout(0.25)(x)

    

    x = Flatten()(x)

    

    x = Dense(128, activation='relu', name='fc0')(x)

    

    x = Dropout(0.5)(x)

    

    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    

    return Model(inputs = x_input, outputs = x, name='Fashion_MNIST')
input_shape = (img_rows, img_cols, 1)



fashionmodel = model(input_shape)

fashionmodel.summary()
fashionmodel.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
history = fashionmodel.fit(x_train,

               y_train,

               epochs=epochs,

               batch_size=batch_size)
print(history.history.keys())



# summarize history for accuracy

plt.plot(history.history['acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train'], loc='upper left')

plt.show()