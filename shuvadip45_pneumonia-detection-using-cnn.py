import numpy as np

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Conv2D, Dropout, Dense, MaxPool2D
datagen = ImageDataGenerator(rescale=1./255)

training_set = datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/train',

        target_size=(128, 128),

        batch_size=32,

        class_mode='binary')

validation_set = datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/val',

        target_size=(128, 128),

        batch_size=10,

        class_mode='binary')
model = Sequential([Conv2D(64, input_shape = [128,128,3], kernel_size = 5, activation = 'relu'),                  

                    Conv2D(64, kernel_size = 5, activation = 'relu'),

                    Dropout(0.25), 

                    Flatten(),

                    Dense(units = 512, activation='relu'), 

                    Dense(units = 128, activation='relu')])

model.add(Dense(units = 1, activation='sigmoid'))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

print(training_set.class_indices)

model.fit(x = training_set, validation_data = validation_set, epochs = 32)
import glob

from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_paths = glob.glob("../input/chest-xray-pneumonia/chest_xray/test/NORMAL/*.jpeg")

imgs = [load_img(img_path,target_size=(128, 128)) for img_path in img_paths]

img_array = np.array([img_to_array(img) for img in imgs])

normal=[]

for image in img_array:

    image = image.reshape(-1,128,128,3)

    normal.append(model.predict(image)[0][0]<0.5)

print("Normal Accuracy: %f"%(normal.count(True)/len(normal)))

img_paths = glob.glob("../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/*.jpeg")

imgs = [load_img(img_path,target_size=(128, 128)) for img_path in img_paths]

img_array = np.array([img_to_array(img) for img in imgs])

pneumonia=[]

for image in img_array:

    image = image.reshape(-1,128,128,3)

    pneumonia.append(model.predict(image)[0][0]>0.5)

print("Pneumonia accuracy: %f"%(pneumonia.count(True)/len(pneumonia)))