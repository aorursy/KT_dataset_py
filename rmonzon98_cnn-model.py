import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, sys

from IPython.display import display

from IPython.display import Image as _Imgdis

from PIL import Image

import numpy as np

from time import time

from time import sleep

from subprocess import check_output

from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



print(os.listdir("../input/rsna-bone-age"))

print(check_output(["ls", "../input/rsna-bone-age"]).decode("utf8"))
folder = "../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset"

folder_test = "../input/rsna-bone-age/boneage-test-dataset/boneage-test-dataset"



onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

onlyfiles_test = [f for f in os.listdir(folder_test) if os.path.isfile(os.path.join(folder_test, f))]

 

print("Se recompílaron {0} imagenes del folder de training set".format(len(onlyfiles)))

print("Imagen de ejemplo: ")



print(onlyfiles[45])

display(_Imgdis(filename=folder + "/" + onlyfiles[45], width=240, height=320))

    

train_files = []

test_files = []



for _file in onlyfiles:

    train_files.append(_file)

print("Hay %d en el array de training" % len(train_files))

img_df = pd.DataFrame(data = train_files, index=None, columns = None)

csv_df = pd.read_csv("../input/rsna-bone-age/boneage-training-dataset.csv")

df_train = pd.concat([img_df,csv_df],axis = 1)

df_train = df_train.rename(index=str, columns={0: "file"})



for _file in onlyfiles_test:

    test_files.append(_file)

print("Hay %d en el array de test" % len(test_files))

img_df_test = pd.DataFrame(data = test_files, index=None, columns = None)

csv_df_test = pd.read_csv("../input/rsna-bone-age/boneage-test-dataset.csv")

df_test = pd.concat([img_df_test,csv_df_test],axis = 1)

df_test = df_test.rename(index=str, columns={0: "file"})



#print (df_test)
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator



model = Sequential()



model.add(Convolution2D(filters = 32, 

                        kernel_size = (3, 3),

                        input_shape = (240, 320, 3),

                        activation = 'relu'))



model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Convolution2D(32, 3, 3, activation = 'relu'))



model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())



model.add(Dense(units = 128, activation = 'relu'))



model.add(Dense(units = 1, activation = 'sigmoid'))



model.compile(optimizer = 'adam' ,

              loss = 'binary_crossentropy', 

              metrics = ['accuracy'])



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.1, 

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



folder = "../input/rsna-bone-age/boneage-training-dataset"

folder_test = "../input/rsna-bone-age/boneage-test-dataset"

training_set = train_datagen.flow_from_directory(folder,

                                                 target_size = (240, 320),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')



test_set = test_datagen.flow_from_directory(folder_test,

                                            target_size = (240, 320),

                                            batch_size = 32,

                                            class_mode = 'categorical')



model.fit_generator(training_set,

                    steps_per_epoch = 100,

                    epochs = 15,

                    validation_data = test_set,

                    validation_steps = 20)