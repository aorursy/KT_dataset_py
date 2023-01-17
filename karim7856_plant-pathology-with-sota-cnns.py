!pip install efficientnet
#Import libraries to handle data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import cv2
df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

df.sample(3)
path_prefix = "/kaggle/input/plant-pathology-2020-fgvc7/images/"

def get_image_path(image_id):

    return path_prefix + image_id + ".jpg"
cols = [df[(df["healthy"] == 1)]["healthy"].count()/df["healthy"].count(), 

     df[(df["multiple_diseases"] == 1)]["multiple_diseases"].count()/df["multiple_diseases"].count(),

     df[(df["rust"] == 1)]["rust"].count()/df["rust"].count(),

     df[(df["scab"] == 1)]["scab"].count()/df["scab"].count()]

labels = ["healthy", "multiple_diseases", "rust", "scab"]



plt.pie(cols, labels=labels)

plt.show()
healthy_img = cv2.imread(get_image_path(df[(df["healthy"] == 1)].iloc[0].image_id))

healthy_img = cv2.cvtColor(healthy_img, cv2.COLOR_BGR2RGB)



multi_diseas_img = cv2.imread(get_image_path(df[(df["multiple_diseases"] == 1)].iloc[0].image_id))

multi_diseas_img = cv2.cvtColor(multi_diseas_img, cv2.COLOR_BGR2RGB)



scab_img = cv2.imread(get_image_path(df[(df["scab"] == 1)].iloc[0].image_id))

scab_img = cv2.cvtColor(scab_img, cv2.COLOR_BGR2RGB)



rust_img = cv2.imread(get_image_path(df[(df["rust"] == 1)].iloc[0].image_id))

rust_img = cv2.cvtColor(rust_img, cv2.COLOR_BGR2RGB)





fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,6))



ax1.imshow(healthy_img/255.0)

ax1.set_title("Healthy")



ax2.imshow(multi_diseas_img/255.0)

ax2.set_title("multiple_diseases")



ax3.imshow(scab_img/255.0)

ax3.set_title("scab")



ax4.imshow(rust_img/255.0)

ax4.set_title("rust")



plt.show()
#Canny Edge detection

#this detects the leaf in the image and crops the image to the size of the leaf

def get_cropped_canny_image(img):

    edges = cv2.Canny(img,100,100)

    edge_coordinates = []

    for i in range(edges.shape[0]):

        for j in range(edges.shape[1]):

            if edges[i,j] != 0:

                edge_coordinates.append((i,j))

    bottom = edge_coordinates[np.argsort([coordinate[0] for coordinate in edge_coordinates])[0]][0]

    top = edge_coordinates[np.argsort([coordinate[0] for coordinate in edge_coordinates])[-1]][0]

    left = edge_coordinates[np.argsort([coordinate[1] for coordinate in edge_coordinates])[0]][1]

    right = edge_coordinates[np.argsort([coordinate[1] for coordinate in edge_coordinates])[-1]][1]

    new_img = img[bottom:top, left:right]

    new_img = cv2.resize(new_img, (img.shape[1],img.shape[0]))

    return new_img

plt.imshow(get_cropped_canny_image(healthy_img))

plt.show()
IMAGE_SIZE = (128,128)
data = [] #empty array to hold result



#load and resize image

for im in df["image_id"]:

    img = cv2.imread(get_image_path(im))

    img = cv2.resize(img, IMAGE_SIZE)

    data.append(img)

print("Loaded and Resized")



data_len = len(data)

#crop image and append to result (augment data)

for im in range(data_len):

    data.append(get_cropped_canny_image(data[im]))



print("cropped")



data_len = len(data)

#blur the whole new data the original and cropped

for im in range(data_len):

    data.append(cv2.blur(data[im], (80,80)))



len(data)
#repeat the original dataframe 4 times to have the same shape as the images

df = pd.concat([df]*4)

len(df)
X = np.array(data)/255.0

y = df.drop("image_id", axis=1)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.applications.resnet_v2 import ResNet152V2

import efficientnet.tfkeras as efn

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8)

#free up sum memory

del X

del y

del df

del data
checkpoint = ModelCheckpoint(

    "/kaggle/working/EfnNetB7", monitor='val_accuracy', verbose=1, save_best_only=True,

    save_weights_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1, min_delta=0.001, mode='min')
datagen = ImageDataGenerator(vertical_flip=True,horizontal_flip=True)

datagen.fit(X_train, augment=True)
my_efficientNet_base = efn.EfficientNetB7(include_top=False, input_shape=(128,128,3), weights='imagenet')

my_efficientNetB7 = Sequential( [my_efficientNet_base, GlobalAveragePooling2D(), Dense(4,activation='softmax')] )

my_efficientNetB7.compile(optimizer=tf.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
history = my_efficientNetB7.fit(datagen.flow(X_train,y_train.values, batch_size=32), 

                                steps_per_epoch=len(X_train)/32,epochs=25, 

                                validation_data=(X_test,y_test.values), callbacks=[reduce_lr])