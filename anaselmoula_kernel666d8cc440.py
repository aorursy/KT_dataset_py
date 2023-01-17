# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras
import cv2

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
path_train_vehicles = "../input/train/train/vehicles"

#list of vehicles
vehicles = os.listdir("../input/train/train/vehicles")
nb_train_vehicles = len(vehicles)
#vehicles labels

train_vehicles_labels =  np.ones(nb_train_vehicles)

gray_image = cv2.imread(path_train_vehicles+'/'+str(vehicles[int(np.random.random()*nb_train_vehicles)]))
#gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.show()

#down sizing makes the image blury
#resized_image = cv2.resize(gray_image,(28,28))
#print(resized_image.shape)
#plt.imshow(resized_image, cmap='gray')
#plt.show()


path_train_non_vehicles = "../input/train/train/non-vehicles"

#list of non-vehicles
non_vehicles = os.listdir("../input/train/train/non-vehicles")
nb_train_non_vehicles = len(non_vehicles)
#non_vehicles labels
train_non_vehicles_labels =  np.zeros(nb_train_non_vehicles)
                                      
gray_image = cv2.imread(path_train_non_vehicles+'/'+str(non_vehicles[int(np.random.random()*nb_train_non_vehicles)]))
#gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.show()

#preparing labels
train_labels = np.concatenate((train_vehicles_labels,train_non_vehicles_labels))
nb_data = len(train_labels)

#preparing images
train_data =[]
for i in range(len(vehicles)):
    train_data.append(cv2.imread(path_train_vehicles + '/' + str(vehicles[i])))
    
for i in range(len(non_vehicles)):
    train_data.append(cv2.imread(path_train_non_vehicles + '/' + str(non_vehicles[i])))
image_nb = int(np.random.random() * nb_data)

print("label: ",train_labels[image_nb])

plt.imshow(train_data[image_nb], cmap='gray')
plt.show()
                                                                        # validation data preparation
#vehicles:
path_val_vehicles = "../input/val/val/vehicles"

#list of vehicles
val_vehicles = os.listdir("../input/val/val/vehicles")
nb_val_vehicles = len(val_vehicles)
#vehicles labels
val_vehicles_labels =  np.ones(nb_val_vehicles)

#non-vehicles:
path_val_non_vehicles = "../input/val/val/non-vehicles"
#list of non-vehicles
val_non_vehicles = os.listdir("../input/val/val/non-vehicles")
nb_val_non_vehicles = len(val_non_vehicles)
#vehicles labels
val_non_vehicles_labels =  np.zeros(nb_val_non_vehicles)

#preparing labels
val_labels = np.concatenate((val_vehicles_labels,val_non_vehicles_labels))
nb_val_data = len(val_labels)

#preparing images
val_data =[]
for i in range(len(val_vehicles)):
    val_data.append(cv2.imread(path_val_vehicles + '/' + str(val_vehicles[i])))
    
for i in range(len(val_non_vehicles)):
    val_data.append(cv2.imread(path_val_non_vehicles + '/' + str(val_non_vehicles[i])))
train_data = np.reshape(train_data, (len(train_data), 64,64,3))
#train_labels = keras.utils.to_categorical(train_labels)

val_data = np.reshape(val_data, (len(val_data), 64,64,3))
#val_labels = keras.utils.to_categorical(val_labels)

CNN = keras.models.Sequential()

CNN.add(keras.layers.Conv2D(12, (3,3), activation = 'elu', input_shape = (64,64,3), padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(24, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(36, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(48, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
#CNN.add(keras.layers.Conv2D(60, (3,3), activation = 'elu', padding='same'))
#CNN.add(keras.layers.MaxPooling2D((2,2)))
#CNN.add(keras.layers.Conv2D(72, (3,3), activation = 'elu', padding='same'))
#CNN.add(keras.layers.MaxPooling2D((2,2)))

CNN.add(keras.layers.Dropout(0.25))

CNN.add(keras.layers.Flatten())
CNN.add(keras.layers.Dense(100, activation = 'elu'))
CNN.add(keras.layers.Dense(1, activation = 'sigmoid'))

CNN.summary()
CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights', save_best_only=True)


train_gen = ImageDataGenerator(rescale=1.0/255,
                              width_shift_range=0.2,
                              height_shift_range=0.2)

val_gen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_gen.flow(train_data,train_labels, batch_size = 64)
val_generator = val_gen.flow(val_data,val_labels, batch_size = 64)

history = CNN.fit_generator(train_generator, steps_per_epoch = len(train_data)/64, validation_data = val_generator, validation_steps = 654, epochs = 50, callbacks = [checkpointer])

CNN.load_weights('weights')
test_data = os.listdir("../input/test/test")
nb_test_data = len(test_data)
#preparing images
X_test_data =[]
for k in range(nb_test_data):
    X_test_data.append(cv2.imread('../input/test/test'+'/'+str(test_data[k])))
 

X_test_data = np.reshape(X_test_data, (nb_test_data, 64,64,3))/255.0
predictions = CNN.predict(X_test_data)
pd.DataFrame({"Id": list(range(1,nb_test_data+1)), "is_car": np.argmax(predictions, axis=1)}).to_csv('submission_test_file.csv', index=False, header=True)
for i in range(len(test_data)):
    if predictions[i]>0.5: pass
       # print(i+1,": vehicle")
    else: pass
        #print(i+1,": non_vehicle")
score = CNN.evaluate(X_test_data, predictions, verbose=1)
print(score)
from sklearn.metrics import log_loss
import pandas as pd

sub = pd.read_csv('submission_test_file.csv')
 
