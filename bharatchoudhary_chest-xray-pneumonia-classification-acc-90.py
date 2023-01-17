# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow import keras
import kerastuner
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import keras.backend as k
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
k.image_data_format()
Data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/'
Train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'
Test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/'
Valid_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/'
Train_generator = ImageDataGenerator(width_shift_range=0.2, 
                                     height_shift_range=0.2, 
                                     shear_range=0.2, 
                                     zoom_range=0.4,
                                     rotation_range= 30,
                                     horizontal_flip=True,
                                    fill_mode = 'nearest',
                                    rescale = 1/255)

Valid_generator = ImageDataGenerator(rescale = 1/255 )

Test_generator = ImageDataGenerator(rescale= 1/255)
x = load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-0490-0001.jpeg')
x = img_to_array(x)
x = x.reshape((1,) + x.shape)
i = 0
for batch in Train_generator.flow(x, batch_size=1,
                                  save_to_dir='/kaggle/working/',
                        save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break 
Train_data = Train_generator.flow_from_directory(Train_dir, 
                                                 batch_size=32,
                                                 target_size=(160,160), 
                                                 classes=['NORMAL','PNEUMONIA'],
                                                 class_mode='binary')

Valid_data = Valid_generator.flow_from_directory(Valid_dir,
                                                 batch_size=32,
                                                target_size=(160,160), 
                                                classes=['NORMAL','PNEUMONIA'],
                                                class_mode='binary')
Test_data = Test_generator.flow_from_directory(Test_dir,
                                               batch_size=39,shuffle = False,
                                              target_size=(160,160), 
                                              classes=['NORMAL','PNEUMONIA'],
                                              class_mode='binary')
Train_data.class_indices
data, label = next(Test_data)
plt.figure(figsize=(10,10))
for i in range(0,9):
    plt.subplot(330+1+i)
    plt.imshow(data[i])
    plt.grid(False)
plt.show()
label
batch_size = 32
print('\n---\tbatch size is\t\t{}'.format(batch_size))
Epochs = 30
print('\n---\tEpoch number is \t{}'.format(Epochs))
num_of_train_samples = len(Train_data)*32
print('\n---\tNum of train sample\t{}'.format(num_of_train_samples))
num_of_valid_samples = len(Valid_data)*32
print('\n---\tNum of valid samples\t{}'.format(num_of_valid_samples))
num_of_test_samples = len(Test_data)*32
print('\n---\tNum of test samples\t{}'.format(num_of_test_samples))
print('Number of images in Normal Category is {}'.format(len(os.listdir(Train_dir+'NORMAL'))))
print('\n************\n')
print('percentages of Normal Category is {}'. format(len(os.listdir(Train_dir+'NORMAL'))/(len(os.listdir(Train_dir+'NORMAL'))+len(os.listdir(Train_dir+'PNEUMONIA')))* 100))
print('\n-------------\n')
print('Number of images in Pneumonia Category is {}'.format(len(os.listdir(Train_dir+'PNEUMONIA'))))
print('\n************\n')
print('percentages of Pneumonia Category is {}'. format(len(os.listdir(Train_dir+'PNEUMONIA'))/(len(os.listdir(Train_dir+'NORMAL'))+len(os.listdir(Train_dir+'PNEUMONIA')))* 100))
print('Number of images in Normal Category is {}'.format(len(os.listdir(Valid_dir+'NORMAL'))))
print('\n************\n')
print('percentages of Normal Category is {}'. format(len(os.listdir(Valid_dir+'NORMAL'))/(len(os.listdir(Valid_dir+'NORMAL'))+len(os.listdir(Valid_dir+'PNEUMONIA')))* 100))
print('\n-------------\n')
print('Number of images in Pneumonia Category is {}'.format(len(os.listdir(Valid_dir+'PNEUMONIA'))))
print('\n************\n')
print('percentages of Pneumonia Category is {}'. format(len(os.listdir(Valid_dir+'PNEUMONIA'))/(len(os.listdir(Valid_dir+'NORMAL'))+len(os.listdir(Valid_dir+'PNEUMONIA')))* 100))
print('Number of images in Normal Category is {}'.format(len(os.listdir(Test_dir+'NORMAL'))))
print('\n************\n')
print('percentages of Normal Category is {}'. format(len(os.listdir(Test_dir+'NORMAL'))/(len(os.listdir(Test_dir+'NORMAL'))+len(os.listdir(Test_dir+'PNEUMONIA')))* 100))
print('\n-------------\n')
print('Number of images in Pneumonia Category is {}'.format(len(os.listdir(Test_dir+'PNEUMONIA'))))
print('\n************\n')
print('percentages of Pneumonia Category is {}'. format(len(os.listdir(Test_dir+'PNEUMONIA'))/(len(os.listdir(Test_dir+'NORMAL'))+len(os.listdir(Test_dir+'PNEUMONIA')))* 100))

def build_model(img_rows, img_cols):
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(img_rows,img_cols,3),activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32,(3,3),activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model
history = build_model(160,160)
history.fit_generator(Train_data,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=12, initial_epoch=10,
                    validation_data=Test_data,
                    validation_steps=num_of_test_samples // batch_size)
history.evaluate_generator(Test_data,workers = -1)[1]*100
Y_pred = history.predict_generator(Valid_data)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(Valid_data.classes, y_pred))
print('Classification Report')
target_names = ['NORMAL','PNEUMONIA']
print(classification_report(Valid_data.classes, y_pred, target_names=target_names))
history.save('chest-xray')
# list all data in history
#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history.history['accuracy'])
plt.plot(history.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history.history['loss'])
plt.plot(history.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
