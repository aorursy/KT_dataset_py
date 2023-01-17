import numpy as np

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
EPOCHS = 25

INIT_LR = 1e-3 #?

BS = 32 #?

default_image_size = tuple((256, 256))

image_size = 0



width=256

height=256

depth=3 #?
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

    

# https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
image_list, label_list = [], []

directory_root = '../input/plantvillage-tomato-leaf-dataset'



try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    #listdir() returns a list containing the names of the entries in the directory given by path. 

    

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        print(f"[INFO] Processing {plant_folder} ...")

        for disease_folder in plant_disease_folder_list :

            # remove .DS_Store from list

            if disease_folder == ".DS_Store" :

                plant_disease_folder_list.remove(disease_folder)



        for plant_disease_folder in plant_disease_folder_list:

            print(f"    [INFO] Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

                if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:200]:

                print(f"           [INFO] Processing {image} ...")

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory))

                    label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
image_size = len(image_list)
#to convert label to numerical value

#the Binary targets transform to a column vector



label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)

listnow=[]



#pickle.load(open( "label_transform.pkl", "rb" ))



#print(image_labels[541])

#print(label_binarizer.classes_)

#print(image_labels)



#arr=np.array(label_binarizer.inverse_transform(label_binarizer.fit_transform([8,1,2,3,1,4])))

#for i in arr:

#    print(i)

# for i in range(len(image_labels)):

#     for j in range(len(image_labels[i])):

#         print(image_labels[j])
print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 255.0

print("[INFO] Spliting data to train, test")

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
model = Sequential()

inputShape = (height, width, depth)

chanDim = -1

if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

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

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network

print("[INFO] training network...")
history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=25, verbose=1

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

# from keras.preprocessing.image import load_img

# i='../input/checkitnow/tamato_leaf3.jpg'

# arr=[]

# img = load_img(i,target_size=(224,224))

# # np_image_list = np.array(image_list, dtype=np.float16) / 225.0

# arr.append(convert_image_to_array(i))

# np_image_list = np.array(image_list, dtype=np.float16) / 225.0

# prediction= model.predict()

pickle.dump(model,open('cnn_model.pkl', 'wb'))


import numpy as np

import pickle

import cv2

# from keras.preprocessing.image import imgtoarray

# tf.keras.preprocessing.image.img_to_array(

#     img, data_format=None, dtype=None

# )



defaultimagesize = tuple((256, 256))





modelpickle = open(r'./cnn_model.pkl', 'rb')

model = pickle.load(modelpickle)



labelencoder = open(r'./label_transform.pkl', 'rb')

labeltransformer = pickle.load(labelencoder)



imgpath=   "../input/checkitnow/tomato_leaf_curl.jpg"



imar = convert_image_to_array(imgpath)

npimagelist = np.array([imar], dtype=np.float16) / 255.0

pred = model.predict(npimagelist)

#print(pred)



print(labeltransformer.inverse_transform(pred))
from PIL import Image

fileobject = open(r'./cnn_model.pkl', 'rb')

model = pickle.load(fileobject)



imgpath= '../input/checkitnow/tomato_leaf_curl.jpg'

Image.open(imgpath)

imar = convert_image_to_array(imgpath)

npimagelist = np.array([imar], dtype=np.float16) / 255.0 

PREDICTEDCLASSES2 = model.predict_classes(npimagelist) 

print(PREDICTEDCLASSES2)

print(label_binarizer.classes_[PREDICTEDCLASSES2])

img=Image.open(imgpath)

img