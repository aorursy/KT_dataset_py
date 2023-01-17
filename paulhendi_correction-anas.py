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
vehicles = os.listdir("../input/train/train/vehicles") # list of images name
nb_train_vehicles = len(vehicles)

#vehicles labels
train_vehicles_labels =  np.ones(nb_train_vehicles) # Always 1 => that's good

gray_image = cv2.imread(path_train_vehicles+'/'+str(vehicles[int(np.random.random()*nb_train_vehicles)]),0) # Why gray ? unfortunately you are loosing information ...
plt.imshow(gray_image, cmap='gray')
plt.show()
path_train_non_vehicles = "../input/train/train/non-vehicles"

#list of non-vehicles
non_vehicles = os.listdir("../input/train/train/non-vehicles") # list of images name
nb_train_non_vehicles = len(non_vehicles)

#non_vehicles labels
train_non_vehicles_labels =  np.zeros(nb_train_non_vehicles) # Always 0 => Also good
                
gray_image = cv2.imread(path_train_non_vehicles+'/'+str(non_vehicles[int(np.random.random()*nb_train_non_vehicles)]),0)
plt.imshow(gray_image, cmap='gray')
plt.show()
#preparing labels
train_labels = np.concatenate((train_vehicles_labels,train_non_vehicles_labels))
nb_data = len(train_labels)

#preparing images
train_data =[]
for i in range(len(vehicles)):
    train_data.append(cv2.imread(path_train_vehicles + '/' + str(vehicles[i]),0))
    
for i in range(len(non_vehicles)):
    train_data.append(cv2.imread(path_train_non_vehicles + '/' + str(non_vehicles[i]),0))
    
# Okay but this is usually not good cause with bigger images it would require more ram space than the computer actually have. Instead a generator should be used
# to read images at each learning step.
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
    val_data.append(cv2.imread(path_val_vehicles + '/' + str(val_vehicles[i]),0))
    
for i in range(len(val_non_vehicles)):
    val_data.append(cv2.imread(path_val_non_vehicles + '/' + str(val_non_vehicles[i]),0))
    
    
    
# Okay basically the same thing with the val folder
# Actually there is a magic Keras generator function that handle all this quite well.
train_data = np.reshape(train_data, (len(train_data), 64,64,1))
print(train_labels) # dimension : n_data * 1
train_labels = keras.utils.to_categorical(train_labels) 
print(train_labels) # dimension : n_data * 2
# Nooooooooooo ..... Not with two classes. You correctly initialized it before : 1 = vehicle and 0 = non vehicle. No need to hot encode with only 2 classes. 
# It doesn't mean it won't work but we could have one neuron instead of two and besides the prediction is easier to understand.

val_data = np.reshape(val_data, (len(val_data), 64,64,1))
val_labels = keras.utils.to_categorical(val_labels)

CNN = keras.models.Sequential()

CNN.add(keras.layers.Conv2D(12, (3,3), activation = 'elu', input_shape = (64,64,1), padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(24, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(36, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(48, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(60, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))
CNN.add(keras.layers.Conv2D(72, (3,3), activation = 'elu', padding='same'))
CNN.add(keras.layers.MaxPooling2D((2,2)))

CNN.add(keras.layers.Dropout(0.1))

CNN.add(keras.layers.Flatten())
CNN.add(keras.layers.Dense(100, activation = 'elu'))
CNN.add(keras.layers.Dense(2, activation = 'softmax')) # softmax is indeed the appropriate function in this case, but with 1 neuron, sigmoid is the way to go.

CNN.summary()
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])  # Okay (with one neuron : 'binary_crossentropy')
# all those steps below are good, yes ! 
# However like I said before, a magic Keras function could have been used instead : flow_from_directory =) 

checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights', save_best_only=True) # A good idea 


train_gen = ImageDataGenerator(rescale=1.0/255,
                              width_shift_range=0.2,
                              height_shift_range=0.2)

val_gen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_gen.flow(train_data,train_labels, batch_size = 64)
val_generator = val_gen.flow(val_data,val_labels, batch_size = 64)

# validation_steps should be nb_val_data/64 like you did for the train generator. Maybe you pasted this from my code, in that case my mistake.
history = CNN.fit_generator(train_generator, steps_per_epoch = len(train_data)/64, validation_data = val_generator, validation_steps = 654, epochs = 50, callbacks = [checkpointer])

CNN.load_weights('weights')
# You should always check your results ! Taking your submission file, your score is 16.2653128485. 
# But looking at your learning phase everything went well, the score shouldn't be that high. 
# To be sure that you predictions are good you can check the prediction yourself since it's not that hard for us to say if there's a car.
# In image 2, the prediction clearly says (0 1) there is a car but when looking at the picture you can see that there is no car.
# Finally your mistake here is that the list of test images wasn't not sorted, not by default, by os.listdir

test_data = os.listdir("../input/test/test")
print(test_data) # unfortunately not sorted, this explains why you have a high score when you shouldn't
test_data_sorted = [str(i) +".png" for i in range(1, nb_test_data+1)]
nb_test_data = len(test_data)

#preparing images
X_test_data =[]
X_test_data_sorted =[]
for k in range(nb_test_data):
    X_test_data.append(cv2.imread('../input/test/test'+'/'+str(test_data[k]),0))
    X_test_data_sorted.append(cv2.imread('../input/test/test'+'/'+str(test_data_sorted[k]),0))
 

X_test_data = np.reshape(X_test_data, (nb_test_data, 64,64,1))/255.0  # Okay
X_test_data_sorted = np.reshape(X_test_data_sorted, (nb_test_data, 64,64,1))/255.0
predictions = CNN.predict(X_test_data)
predictions2 = CNN.predict(X_test_data_sorted)
# The instruction below is actually not correct, I guess you pasted it from the other kernel which is okay but it doesn't suit our format here
#pd.DataFrame({"ImageId": list(range(1,nb_test_data+1)), "label": np.argmax(predictions, axis=1)}).to_csv('submission_test_file.csv', index=False, header=True)

pd.read_csv("../input/submission_sample.csv").head()
# Instead, this one is more appropriate looking at the submission_sample file provided with the data 
pd.DataFrame({"id": list(range(1,nb_test_data+1)), "is_car": np.argmax(predictions2, axis=1)}).to_csv('submission_test_file.csv', index=False, header=True)
