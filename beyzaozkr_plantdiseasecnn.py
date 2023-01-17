from os import listdir

import cv2

import numpy as np

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras import layers

from keras import models

from keras.layers.convolutional import Conv2D

from keras import optimizers

import matplotlib.pyplot as plt

from keras.optimizers import Adam

from keras.models import Sequential

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing import image
default_image_size = tuple((256, 256))
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None:

            image =cv2.resize(image, default_image_size)

            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

            return img_to_array(image)

        else:

            return  np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
directory_root = '../input/plantdiseasedataset2/plantdiseasedataset'
X=[]

Y=[]

try:

    print("Fotoğraflar yükleniyor ...")

    root_dir = listdir(directory_root)



    for color_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{color_folder}")



        for plant_disease_folder in plant_disease_folder_list:

            print(f"{plant_disease_folder} Yükleniyor ...")

            plant_disease_image_list = listdir(f"{directory_root}/{color_folder}/{plant_disease_folder}/")



            for image in plant_disease_image_list[:200]:

                image_directory = f"{directory_root}/{color_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    X.append(convert_image_to_array(image_directory))

                    Y.append(plant_disease_folder)

    print("Fotoğraf yükleme tamamlandı")  

except Exception as e:

    print(f"Error : {e}")
Y_binarizer = LabelBinarizer()

Y = Y_binarizer.fit_transform(Y)

print(Y_binarizer.classes_)
np_X = np.array(X,dtype=np.float16) / 255.0
x_train, x_test, y_train, y_test = train_test_split(np_X, Y, test_size=0.2, random_state = 42)

print("Veri train, test olarak ayrılıyor")
dataAugmentation = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2,

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
model = models.Sequential()

inputShape = (256, 256, 1)

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=inputShape))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(33, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(), metrics=['accuracy'])
history = model.fit_generator(

    dataAugmentation.flow(x_train, y_train, batch_size=32),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) / 32,

    epochs=30, verbose=1

    )
model_json = model.to_json()

with open('model.json', 'w') as json_file:

    json_file.write(model_json)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
#model ve mimariyi kaydet

model.save("model.h5")

print("Saved model to disk(h5)")
# modeli pickle ile kaydet

import pickle

pickle.dump(model,open('plant_disease_model.pkl', 'wb'))

print("[INFO] Model kaydediliyor...")

# ağırlıkları kaydet

model.save_weights("model_weights.h5")

print("Saved model to disk (h5)")