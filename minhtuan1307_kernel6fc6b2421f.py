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
import keras

from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.optimizers import Adam



from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



import time

import numpy as np

import matplotlib.pyplot as plt
num_classes = 7   # angry, disgust, fear, happy, neutral, sad, and surprise



trainingset = np.loadtxt('/kaggle/input/fer2013_training_onehot.csv', delimiter=',')

testingset = np.loadtxt('/kaggle/input/fer2013_publictest_onehot.csv', delimiter=',')

n_inputs = 2304

n_classes = 7

img_dim = 48



x_training = trainingset[:, 0:n_inputs]

y_training = trainingset[:, n_inputs:n_inputs + n_classes]



x_testing = testingset[:, 0:n_inputs]

y_testing = testingset[:, n_inputs:n_inputs + n_classes]



x_training = x_training.reshape(x_training.shape[0], 48, 48)

x_training = np.expand_dims(x_training, axis=4)



x_testing = x_testing.reshape(x_testing.shape[0], 48, 48)

x_testing = np.expand_dims(x_testing, axis=4)
testingset
from keras.regularizers import l2

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization

from keras.models import Sequential
# 64x64 portray image

input_image = Input(shape=(48, 48, 1), name='Input')

model = Sequential()

    

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)))

model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Dropout(0.5))

    

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

    

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

    

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

    

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# print model summary

model.summary()

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# declare learning rate, loss function, and model metric

loss = 'categorical_crossentropy'

lr = 0.001

model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=['accuracy'])



# train the model

batch_size = 64

epochs = 100



starting_time = time.time()

history = model.fit(x_training, y_training,

                    validation_data=(x_testing, y_testing),

                    batch_size=batch_size,

                    epochs=epochs)

print('> training time is %.4f minutes' % ((time.time() - starting_time)/60))
score = model.evaluate(x_testing, y_testing)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def get_emotion(ohv):

    indx = np.argmax(ohv)

        

    if indx == 0:

        return 'angry'

    elif indx == 1:

        return 'disgust'

    elif indx == 2:

        return 'fear'

    elif indx == 3:

        return 'happy'

    elif indx == 4:

        return 'sad'

    elif indx == 5:

        return 'surprise'

    elif indx == 6:

        return 'neutral'
img_indx = np.uint32(np.random.rand()*(testingset.shape[0] - 1))

sample = x_testing[img_indx, :]

sample = sample.reshape(48, 48)



pred_cls = model.predict(sample.reshape(1, 48, 48, 1))



plt.imshow(sample, cmap='gray')

plt.show()

print('> testing image index: %d\n> true emotion: %s\n> predicted emotion: %s' % (img_indx, get_emotion(y_testing[img_indx, :]), get_emotion(pred_cls)))