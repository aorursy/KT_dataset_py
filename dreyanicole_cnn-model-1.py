import numpy as np # linear algebra

import pickle

import pandas as pd #data handling

import cv2 #opencv functions

from keras import layers

from keras import models

from os import listdir #to access input directories



#modules used for making the model

from keras.models import Sequential

from keras import backend as K

from keras.layers.convolutional import Conv2D

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense



#used for training

from keras.optimizers import Adam



from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.image import ImageDataGenerator #for image preprocessing

from keras.preprocessing.image import img_to_array #turns image to arrays

from sklearn.model_selection import train_test_split #splitting train and test data



import matplotlib.pyplot as plt



EPOCHS = 25

INIT_LR = 1e-3

BS = 32

default_image_size = tuple((256, 256))

image_size = 0

directory_root = '../input/eggplant-leaf-disease/'

width=256

height=256

depth=3
def convert_image_to_array(image_dir, colorspc):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            if colorspc=="rgb":

                image = cv2.resize(image, default_image_size)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                return img_to_array(image)

            elif colorspc =="hsv":

                image = cv2.resize(image, default_image_size) 

                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                return img_to_array(image)

            elif colorspc =="ycc":

                image = cv2.resize(image, default_image_size) 

                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

                return img_to_array(image)

            elif colorspc =="cie":

                image = cv2.resize(image, default_image_size) 

                image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

                return img_to_array(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
image_list, hsv_image_list, ycc_image_list, cie_image_list, label_list = [], [], [], [], []

try:

    print(">> Loading images ...")

    root_dir = listdir(directory_root)

    

    for directory in root_dir :

        # remove .DS_Store from list of root directory (eggplant-leaf-disease)

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    #getting the folder directories inside eggplant-leaf-diseases

    for eggplant_folder in root_dir :

        eggplant_disease_folder_list = listdir(f"{directory_root}/{eggplant_folder}")

        

        for disease_folder in eggplant_disease_folder_list :

            # remove .DS_Store from eggplant_data

            if disease_folder == ".DS_Store" :

                eggplant_disease_folder_list.remove(disease_folder)

        

        #getting the folder directories under eggplant_data

        for eggplant_disease_folder in eggplant_disease_folder_list:

            print(f"[INFO] Processing {eggplant_disease_folder} ...")

            eggplant_disease_image_list = listdir(f"{directory_root}/{eggplant_folder}/{eggplant_disease_folder}/")

                

            for single_eggplant_disease_image in eggplant_disease_image_list :

                # remove .DS_Store from folders inside eggplant_data (eggplant_healthy and eggplant_leafminer)

                if single_eggplant_disease_image == ".DS_Store" :

                    eggplant_disease_image_list.remove(single_eggplant_disease_image)



            #getting image from eggplant_healthy and eggplant_leafminer

            for image in eggplant_disease_image_list[:250]: #picked 100 per folder/class

                image_directory = f"{directory_root}/{eggplant_folder}/{eggplant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory, "rgb"))

                    hsv_image_list.append(convert_image_to_array(image_directory, "hsv"))

                    ycc_image_list.append(convert_image_to_array(image_directory, "ycc"))

                    cie_image_list.append(convert_image_to_array(image_directory, "cie"))

                    label_list.append(eggplant_disease_folder)

    print(">> Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
image_size = len(image_list)

label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb')) #saved the deserialized objects using pickle

n_classes = len(label_binarizer.classes_) #saving the classes created using the binarizer
print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16)/255

np_hsv_image_list = np.array(hsv_image_list, dtype=np.float16)/255

np_ycc_image_list = np.array(ycc_image_list, dtype=np.float16)/255

np_cie_image_list = np.array(cie_image_list, dtype=np.float16)/255
print("[INFO] Spliting data to train, test")

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 

hsv_x_train, hsv_x_test, hsv_y_train, hsv_y_test = train_test_split(np_hsv_image_list, image_labels, test_size=0.2, random_state = 42)

ycc_x_train, ycc_x_test, ycc_y_train, ycc_y_test = train_test_split(np_ycc_image_list, image_labels, test_size=0.2, random_state = 42)

cie_x_train, cie_x_test, cie_y_train, cie_y_test = train_test_split(np_cie_image_list, image_labels, test_size=0.2, random_state = 42)
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
model = Sequential() #initializing model

inputShape = (height, width, depth) 

chanDim = -1 #defualting to channel last

if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1 #but giving the option to channel first

    

#creating first layer of conv=>relu=>pool

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25)) #applying 25% dropout



#creating two sets of (conv=>relu)*2=>pool

#set1

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25)) #applying dropout of 20%

#set2

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



#creating FC=>Relu

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(n_classes))

model.add(Activation("softmax"))

model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# distribution

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=["sparse_categorical_accuracy"])
# train the network

print(">> training network : BGR")



history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=EPOCHS, verbose=1

    )



history_dict = history.history

history_dict.keys()



acc = history.history['sparse_categorical_accuracy']

val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()



#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()



print(">> Calculating model accuracy")

scores = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {scores[1]*100}")





print(">> training network : HSV")

hsv_history = model.fit_generator(

    aug.flow(hsv_x_train, hsv_y_train, batch_size=BS),

    validation_data=(hsv_x_test, hsv_y_test),

    steps_per_epoch=len(hsv_x_train) // BS,

    epochs=EPOCHS, verbose=1

    )



history_dict = hsv_history.history

history_dict.keys()



acc = hsv_history.history['sparse_categorical_accuracy']

val_acc = hsv_history.history['val_sparse_categorical_accuracy']

loss = hsv_history.history['loss']

val_loss = hsv_history.history['val_loss']

epochs = range(1, len(acc) + 1)



#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()



#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()



print(">> Calculating model accuracy")

scores = model.evaluate(hsv_x_test, hsv_y_test)

print(f"Test Accuracy: {scores[1]*100}")



print(">> training network : YCC")

ycc_history = model.fit_generator(

    aug.flow(ycc_x_train, ycc_y_train, batch_size=BS),

    validation_data=(ycc_x_test, ycc_y_test),

    steps_per_epoch=len(ycc_x_train) // BS,

    epochs=EPOCHS, verbose=1

    )



history_dict = ycc_history.history

history_dict.keys()



acc = ycc_history.history['sparse_categorical_accuracy']

val_acc = ycc_history.history['val_sparse_categorical_accuracy']

loss = ycc_history.history['loss']

val_loss = ycc_history.history['val_loss']

epochs = range(1, len(acc) + 1)



#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()



#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()



print(">> Calculating model accuracy")

scores = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {scores[1]*100}")



print(">> training network : CIE")

cie_history = model.fit_generator(

    aug.flow(cie_x_train, cie_y_train, batch_size=BS),

    validation_data=(cie_x_test, cie_y_test),

    steps_per_epoch=len(cie_x_train) // BS,

    epochs=EPOCHS, verbose=1

    )
cie_history_dict = history.history

history_dict.keys()



acc = cie_history.history['sparse_categorical_accuracy']

val_acc = cie_history.history['val_sparse_categorical_accuracy']

loss = cie_history.history['loss']

val_loss = cie_history.history['val_loss']

epochs = range(1, len(acc) + 1)



#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()



#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
print(">> Calculating model accuracy")

scores = model.evaluate(cie_x_test, cie_y_test)

print(f"Test Accuracy: {scores[1]*100}")