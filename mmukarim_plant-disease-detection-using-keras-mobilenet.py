import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense , DepthwiseConv2D

from keras.models import Model

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
EPOCHS = 10

INIT_LR = 1e-3

BS = 32

default_image_size = tuple((256, 256))

image_size = 0

directory_root = '../input/labelled/'

width=256

height=256

depth=3
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            image = cv2.resize(image, default_image_size)   

            return img_to_array(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
image_list, label_list = [], []

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        

        for disease_folder in plant_disease_folder_list :

            # remove .DS_Store from list

            if disease_folder == ".DS_Store" :

                plant_disease_folder_list.remove(disease_folder)



        for plant_disease_folder in plant_disease_folder_list:

            print(f"[INFO] Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

                if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:200]:

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory))

                    label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
image_size = len(image_list)
label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print("[INFO] Spliting data to train, test")

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
#Mobilenet from scratch 

activation = 'relu'

inputs = Input((256, 256, 3))

x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(

            inputs)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



for _ in range(5):

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

    x = BatchNormalization()(x)



    x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(

                x)

    x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)

x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = GlobalAveragePooling2D()(x)

out = Dense(n_classes, activation='softmax')(x)



model = Model(inputs, out)
model.summary()
opt = Adam()

# lr=INIT_LR, decay=INIT_LR / EPOCHS

# distribution

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network

print("[INFO] training network...")
history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=EPOCHS, verbose=1

    )
acc = history.history['acc']

val_acc = history.history['val_acc']

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
print("[INFO] Calculating model accuracy")

scores = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {scores[1]*100}")
# save the model to disk

print("[INFO] Saving model...")

pickle.dump(model,open('cnn_model.pkl', 'wb'))