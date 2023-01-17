# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from keras.preprocessing.image import img_to_array

from os import listdir

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/plantvillage"))



# Any results you write to the current directory are saved as output.
EPOCHS = 25

INIT_LR = 1e-3

BS = 32

default_image_size = tuple((256, 256))

image_size = 0

directory_root = '../input/plantvillage/'

width=256

height=256

depth=3
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None:

            image = cv2.resize(image,default_image_size)

            return img_to_array(image)

        else:

            return np.array([])

    except Exception as e:

        print(f'Error {e}')

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

                plant_disease_folder_list.remove(disease_folder)   # remove file 

                plant_disease_folder_list.remove('Potato___healthy')   # remove file 

                



        for plant_disease_folder in plant_disease_folder_list:

            print(f"[INFO] Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

                if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:350]:

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory))

                    label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
plant_disease_folder_list

image_size = len(image_list) # size of processed images
image_size
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)
import pickle

pickle.dump(label_binarizer,open('label_transform.pkl','wb'))

n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
np_image_list = np.array(image_list,dtype=np.float16) / 255.0
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(np_image_list, image_labels,test_size = 0.2,random_state = 50)
from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(

    rotation_range = 25,

    width_shift_range = 0.1,

    height_shift_range = 0.1,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    fill_mode="nearest"

)
from keras.models import Sequential

from keras import backend as K

from keras.layers import Activation,Conv2D,BatchNormalization,Flatten,Dense,Dropout,MaxPooling2D



model = Sequential()

inputShape = (height,width,depth)

chanDim = -1



if K.image_data_format() == "channels_first":

    inputShape = (depth,height,width)

    chanDim = 1

# convolution

model.add(Conv2D(32,

                (5,5),

                padding = "same",

                input_shape = inputShape))

# activation          

model.add(Activation(activation = 'relu')) 

#pooling

model.add(MaxPooling2D(pool_size = (3,3) ))

#model.summary()



# convolution

model.add(Conv2D(64,

                (5,5),

                padding = "same"

                ))

# activation          

model.add(Activation(activation = 'relu')) 

#pooling

model.add(MaxPooling2D(pool_size = (2,2) ))



# convolution

model.add(Conv2D(64,

                (3,3),

                padding = "same"

                ))

# activation          

model.add(Activation(activation = 'relu')) 

#pooling

model.add(MaxPooling2D(pool_size = (2,2) ))





# convolution

model.add(Conv2D(128,

                (3,3),

                padding = "same"

                ))

# activation          

model.add(Activation(activation = 'relu')) 

#pooling

model.add(MaxPooling2D(pool_size = (2,2) ))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(1024))



model.add(Activation(activation='relu'))



model.add(BatchNormalization())



model.add(Dropout(0.5))



model.add(Dense(n_classes))



model.add(Activation("softmax"))

model.summary()
from keras.optimizers import Adam

opt = Adam(lr = INIT_LR

          ,decay=INIT_LR/EPOCHS)

model.compile(loss="binary_crossentropy",

             optimizer=opt,

             metrics=['accuracy'])

print('[INFO] training the network...')
history = model.fit_generator(

        aug.flow(X_train,y_train,batch_size=BS),

        validation_data=(X_test,y_test),

        steps_per_epoch=len(X_train) // BS,

        epochs=EPOCHS, verbose=1

)
scores = model.evaluate(X_test,y_test)

print(scores)
import pickle

filename = 'model.pkl'

pickle.dump(model.open(filename, 'wb'))
from sklearn.externals import joblib



# Saving a model

joblib.dump(model, 'model.pkl')
from keras.models import load_model



model.save('my_model.h5')  # creates a HDF5 file
print(os.listdir("../input/plantvillage"))
print(os.system('ls'))