# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import np_utils

import numpy as np

from keras.preprocessing.image import array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model

from keras.models import Sequential

from tensorflow.keras import backend, layers, models

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Model

from keras.layers import BatchNormalization

from keras.layers import Conv2D,MaxPooling2D

from keras.layers import Activation, Dense, Flatten, Dropout

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Flatten, Activation
from sklearn.datasets import load_files

train_dat = '/kaggle/input/fruits/fruits-360/Training'

test_dat = '/kaggle/input/fruits/fruits-360/Test'



def load_dataset(path):

    data = load_files(path)

    files = np.array(data['filenames'])

    targets = np.array(data['target'])

    target_labels = np.array(data['target_names'])

    return files,targets,target_labels

    

x_train, y_train,target_labels = load_dataset(train_dat)

x_test, y_test,_ = load_dataset(test_dat)



print('Training set size : ' , x_train.shape[0])

print('Testing set size : ', x_test.shape[0])


y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)



y_train = y_train [2001:4001]

x_train = x_train[2001:4001]

x_test,x_valid = x_test[0:1000],x_test[1001:2000]

y_test,y_valid = y_test[0:1000],y_test[1001:2000]

def convert_image_to_array(files):

    images_as_array=[]

    for file in files:

        images_as_array.append(img_to_array(load_img(file)))

    return images_as_array





x_train = np.array(convert_image_to_array(x_train))



x_valid = np.array(convert_image_to_array(x_valid))



x_test = np.array(convert_image_to_array(x_test))


x_train = x_train.astype('float32')/255

x_valid = x_valid.astype('float32')/255

x_test = x_test.astype('float32')/255
fig = plt.figure(figsize =(30,5))

for i in range(10):

    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])

    ax.imshow(np.squeeze(x_train[i]))
backend.clear_session()

visible = Input(shape=(100,100,3))



T1= layers.Conv2D(100, 100, 3, padding='same', activation='relu') (visible)

T1 = layers.MaxPooling2D(3, padding= 'same') (T1)



T2= layers.Conv2D(100, 100, 3, padding='same', activation='relu') (visible)

T2 = layers.MaxPooling2D(3, padding='same') (T2)



T3 = Dropout(0.5) (visible)

T3 = layers.Conv2D(100, 100, 3, padding='same', activation= 'relu') (T3)

T3 = layers.MaxPooling2D(3, padding='same') (T3)



T4 = layers.Conv2D(100, 100, 3, padding ='same', activation='relu') (visible)

T4 = layers.MaxPooling2D(3, padding='same') (T4)



merge = Concatenate(axis=-1)([T1, T2, T3, T4])



norm = layers.BatchNormalization() (merge)

flat = Flatten() (norm)

drop =Dropout(0.02) (flat)

output = Dense(120, activation='relu') (drop)



model = Model(inputs=visible, outputs=output)



plot_model(model)



model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



history = model.fit(x_train, 

          y_train, 

          epochs = 32, 

          batch_size = 75,  

          validation_data=(x_valid, y_valid),

          verbose = 1)



history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(history_dict['accuracy']) + 1)



plt.plot(epochs, loss_values, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')

plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



test_loss, test_acc = model.evaluate(x_test, y_test)

print('test_acc:', test_acc)


