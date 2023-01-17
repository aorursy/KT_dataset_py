# Import Libraries

import os

import glob

import matplotlib.pyplot as plt

import numpy as np

# Keras API

import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

import pickle 

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

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
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        

        if image is not None:

            image = cv2.resize(image, (256, 256))

            return img_to_array(image)

        else:

            return np.array([])

    except Exception as e:

        print(f"Error: {e}")

        return None

    

directory_root = '/kaggle/input/mohantydata'

image_list, label_list = [], []

max_classes_number=10

classes_count=0

try:

    print("(INFO) Loading images ...")

    root_dir = listdir(directory_root)

    print(root_dir)

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)    

    for plant_disease_folder in root_dir:

        if classes_count > max_classes_number:

            break

        print(f" [INFO] Processing {plant_disease_folder} ...")

        plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")

            

        for single_plant_disease_image in plant_disease_image_list :

            if single_plant_disease_image == ".DS_Store" :

                plant_disease_image_list.remove(single_plant_disease_image)



        for image in plant_disease_image_list[:250]:

            image_directory = f"{directory_root}/{plant_disease_folder}/{image}"

                              

            if image_directory.endswith(".jpg") == True or image_directory .endswith(".JPG") == True:

                image_list.append(convert_image_to_array(image_directory))

                label_list.append(plant_disease_folder)

        

        classes_count += 1

    print(" (INFO) Image loading completed")

except Exception as e:

    print(f"Error : {e}")
label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype = np.float16)
print('[INFO] Splitting data to train and test')

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size = 0.2, random_state = 42)
train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)
# set height and width and color of input image.

img_width,img_height =256,256

input_shape= (img_width,img_height,3)

batch_size = 32

train_generator =train_datagen.flow(x_train, y_train,

                                                   batch_size=batch_size)

test_generator=test_datagen.flow(x_test, y_test,batch_size=batch_size)
# CNN building.

num_classes = 11

model = Sequential()

model.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))   

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(128,activation='relu'))          

model.add(Dense(num_classes,activation='softmax'))

model.summary()
from keras.preprocessing import image

import numpy as np

img1 = image.load_img('/kaggle/input/mohantydata/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG')

plt.imshow(img1);

#preprocess image

img1 = image.load_img('/kaggle/input/mohantydata/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG', target_size=(256, 256))

img = image.img_to_array(img1)

img = img/255.0

img = np.expand_dims(img, axis=0)
import matplotlib.image as mpimg



def visualize(layer):

    fig=plt.figure(figsize=(14,7))

    columns = 8

    rows = 4

    for i in range(columns*rows):

        #img = mpimg.imread()

        fig.add_subplot(rows, columns, i+1)

        plt.axis('off')

        plt.title('filter'+str(i))

        plt.imshow(layer[0, :, :, i], cmap='viridis') # Visualizing in color mode.

    plt.show()
from keras.models import Model

conv2d_1_output = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)

max_pooling2d_1_output = Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_1').output)

conv2d_2_output=Model(inputs=model.input,outputs=model.get_layer('conv2d_2').output)

max_pooling2d_2_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_2').output)

conv2d_3_output=Model(inputs=model.input,outputs=model.get_layer('conv2d_3').output)

max_pooling2d_3_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_3').output)

flatten_1_output=Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)

conv2d_1_features = conv2d_1_output.predict(img)

max_pooling2d_1_features = max_pooling2d_1_output.predict(img)

conv2d_2_features = conv2d_2_output.predict(img)

max_pooling2d_2_features = max_pooling2d_2_output.predict(img)

conv2d_3_features = conv2d_3_output.predict(img)

max_pooling2d_3_features = max_pooling2d_3_output.predict(img)

flatten_1_features = flatten_1_output.predict(img)





visualize(conv2d_1_features)

#visualize(max_pooling2d_1_features)


opt=keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

train=model.fit_generator(train_generator,

                          nb_epoch=28,

                          steps_per_epoch=len(x_train)//batch_size,

                          validation_data=test_generator,

                          verbose=1)
print("[INFO] Calculating model accuracy")

scores = model.evaluate(test_generator)

print(f"Test Accuracy: {scores[1]*100}")



from keras.models import load_model

model.save('crop.h5')