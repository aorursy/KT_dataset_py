# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
img_rows, img_cols = 28, 28

num_classes = 10



train_x = train.values[:, 1:].astype('float32')

train_y = keras.utils.to_categorical(train.label, num_classes)



print(train_x.shape)

print(train_y.shape)



test_x = test.values.astype('float32')



#scale = np.max(train_x)



#train_x /= scale

#test_x /= scale



#mean = np.std(train_x)



#train_x -= mean

#test_x -= mean



train_x /= 255

test_x /= 255



train_x = train_x.reshape(train_x.shape[0], img_rows, img_cols, 1)

test_x = test_x.reshape(test_x.shape[0], img_rows, img_cols, 1)



print(train_x.shape)

print(train_y.shape)
#x, val_x, y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=88)

#print(x.shape)

#print(y.shape)
import matplotlib.pyplot as plt

import random



#extra_train_x = np.empty(train_x.shape)

#extra_train_y = np.empty(train_y.shape)

#for idx in range(train_x.shape[0]):

#    axis = random.randint(0, 1)

#    roll_quantity = 3 * (1 if random.random() < 0.5 else -1)

#    extra_train_x[idx] = np.roll(train_x[idx], roll_quantity, axis=axis)

#    extra_train_y[idx] = train_y[idx]

    

#x = np.concatenate((extra_train_x, train_x), axis=0)

#y = np.concatenate((extra_train_y, train_y), axis=0)



#plt.figure(num='digit',figsize=(9,9))

#for idx in range(0, 9):

#    pixels = extra_train_x[idx]

#    label = extra_train_y[idx]



#    pixels = pixels.reshape((28, 28))

#    num = np.argmax(label, axis=0)



#    plt.subplot(3, 3, idx+1)

    # Plot

#    plt.title('L: {label}'.format(label=num))

#    plt.imshow(pixels)



#plt.show()
models = []



for idx in range(15):

    model = Sequential()



    model.add(Conv2D(32, kernel_size=3,

                     activation='relu',

                     input_shape=(img_rows, img_cols, 1)))

    model.add(BatchNormalization())



    model.add(Conv2D(32, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    

    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))



    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    models.append(model)
def shift(arr, num, axis, fill_value=0.):

    result = np.empty_like(arr)

    if axis == 0:

        if num > 0:

            result[:num, :] = fill_value

            result[num:, :] = arr[:-num, :]

        elif num < 0:

            result[num:, :] = fill_value

            result[:num, :] = arr[-num:, :]

        else:

            result = arr

    elif axis == 1:

        if num > 0:

            result[:, :num] = fill_value

            result[:, num:] = arr[:, :-num]

        elif num < 0:

            result[:, num:] = fill_value

            result[:, :num] = arr[:, -num:]

        else:

            result = arr

        

    return result



def gen_img(arr):

    shif = 4 * (1 if random.random() < 0.5 else -1)

    axis = random.randint(0, 1)

    return shift(arr, shif, axis)



def gen_data(x, y):

    new_x = np.empty(x.shape)

    new_y = np.empty(y.shape)

    for idx in range(x.shape[0]):

        new_x[idx] = gen_img(x[idx])

        new_y[idx] = y[idx]

        

    return new_x, new_y



#sample_x, sample_y = gen_data(train_x, train_y)

#for idx in range(0, 5):

#    img = sample_x[idx].reshape(28,28)

    

#    plt.title('L: {label}'.format(label=np.argmax(sample_y[idx])))

#    plt.imshow(img)

#    plt.show()
es = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=50, restore_best_weights=True)

lrs = keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)



for i in range(0, 0):

    extra_train_x, extra_train_y = gen_data(train_x, train_y)

    x = np.concatenate((extra_train_x, train_x), axis=0)

    y = np.concatenate((extra_train_y, train_y), axis=0)



    models[i].fit(x, y,

              batch_size=128,

              epochs=50,

              validation_split=0.1,

              callbacks=[es, lrs])
es = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=50, restore_best_weights=True)

lrs = keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)



for i in range(0, 15):

    datagen = keras.preprocessing.image.ImageDataGenerator(

            rotation_range=10,  

            zoom_range=0.10,  

            width_shift_range=0.10, 

            height_shift_range=0.10)



    gen_train_x, gen_val_x, gen_train_y, gen_val_y = train_test_split(train_x, train_y, test_size=0.1)



    models[i].fit_generator(datagen.flow(gen_train_x, gen_train_y, batch_size=64),

                        epochs=50,

                        steps_per_epoch=gen_train_x.shape[0]//64,

                        validation_data=(gen_val_x,gen_val_y),

                        callbacks=[lrs])

                    
results = np.zeros((test_x.shape[0], 10))

for i in range(15):

    results = results + models[i].predict(test_x)
predictions = np.argmax(results, axis=1)
submission = pd.DataFrame(data={'ImageId': pd.RangeIndex(start=1, stop=test_x.shape[0]+1), 'Label': predictions})

submission = submission.set_index('ImageId')
submission.to_csv('submission.csv')