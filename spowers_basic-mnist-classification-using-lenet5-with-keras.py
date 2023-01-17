import keras

print("keras running version {}".format(keras.__version__))

from keras import backend as K

print(K.tensorflow_backend._get_available_gpus())
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import backend

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.utils import plot_model

from sklearn.utils import shuffle

import tensorflow as tf

#set the backend so that 1,x,y can be used for the convolution layer format.

backend.set_image_data_format('channels_first')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print("Files available in input directory:")

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

def normalize_img(img):

    return img/255.



def mish(x):

   return x * keras.backend.tanh(keras.backend.softplus(x))
from keras.utils.np_utils import to_categorical



def preprocess_data(X, y=None):

    #reshape from pandas dataframe

    X = X.values

    X = X.reshape(X.shape[0], 1, 28, 28).astype(np.float32)

    #normalize

    X = np.array([normalize_img(img) for img in X])

    #shuffle and encode

    if y is not None:

        y = to_categorical(y)

        X_data, y_data = shuffle(X, y)

    else:

        X_data, y_data = X, None

    return X_data, y_data
training_data = pd.read_csv("../input/train.csv")

x_train, y_train = preprocess_data(training_data.drop(['label'], axis=1) ,training_data['label'])

x_test, y_test  = preprocess_data(pd.read_csv('../input/test.csv'))

#Setup LeNet5 architecture



model = Sequential([

    Conv2D(6, (3, 3), activation=mish, input_shape=(1,28,28)),

    MaxPooling2D((2,2)),

    Conv2D(6, (3, 3), activation=mish),

    #MaxPooling2D((2,2)),

    Dropout(0.5),

    Conv2D(16, (3,3), activation=mish),

    #MaxPooling2D((2,2)),

    Dropout(0.5),

    Conv2D(16, (3,3), activation=mish),

    MaxPooling2D((2,2)),

    Dropout(0.5),    

    Flatten(),

    Dense(120),

    Dense(84),

    Dense(10),

    Activation('softmax')

])



plot_model(model, to_file='model.png')
learning_rate = 3e-2

epochs = 40

decay = learning_rate / epochs

optimizer = keras.optimizers.adam(lr=learning_rate, decay=decay)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_split=0.3) 

test_score = model.evaluate(x_train, y_train)

predicted = model.predict(x_test).argmax(-1)
model.save('mnist.h5')
results = pd.DataFrame([v for v in zip(range(1,len(predicted)+1), predicted)], columns=['ImageId', 'Label'])
results.to_csv('final_results.csv', header=True, index=False)