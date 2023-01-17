#Basics
import numpy as np
import pandas as pd

#Utility
import urllib
import os
import shutil
import time
from PIL import Image

#Vizualization
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import cv2

#MODEL
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout, AveragePooling2D, Concatenate
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)
df.head()
#Checking out annotation structure
df['annotation'][1]
#Make Directory to Store Pictures
os.mkdir("IndianPlates")
#Dictionray For Import
dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["top_x"] = list()
dataset["top_y"] = list()
dataset["bottom_x"] = list()
dataset["bottom_y"] = list()
#Extracting the Image and popualting the Dataset Dictionary
counter = 0
for index, row in df.iterrows():
    img = urllib.request.urlopen(row["content"])
    img = Image.open(img)
    img = img.convert('RGB')
    img.save(f"IndianPlates/licensed_car{counter}.jpeg","JPEG")

    dataset["image_name"].append(f"licensed_car{counter}.jpeg")
    
    data = row["annotation"]
    
    dataset["image_width"].append(data[0]["imageWidth"])
    dataset["image_height"].append(data[0]["imageHeight"])
    dataset["top_x"].append(data[0]["points"][0]["x"])
    dataset["top_y"].append(data[0]["points"][0]["y"])
    dataset["bottom_x"].append(data[0]["points"][1]["x"])
    dataset["bottom_y"].append(data[0]["points"][1]["y"])
    
    counter += 1
print(f"Downloaded {counter} car images.")
#Saved For Future Use
pd.DataFrame(dataset).to_csv("indian_plates.csv", index=False)
df = pd.read_csv("indian_plates.csv")
df.head()
WIDTH = 224
HEIGHT = 224

def display_car_image(index, scale=True, WIDTH=224, HEIGHT=224):
    images_dir = 'IndianPlates/'
    
    img = cv2.imread(images_dir + df['image_name'].iloc[index])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(WIDTH, HEIGHT))

    top_x = int(df['top_x'].iloc[index] * WIDTH)
    top_y = int(df['top_y'].iloc[index] * HEIGHT)
    bottom_x = int(df['bottom_x'].iloc[index] * WIDTH)
    bottom_y = int(df['bottom_y'].iloc[index] * HEIGHT)
    center_x = 20
    center_y = 20

    # adding bounding box to the number plate in the car
    bbox_width = 1
    bbox_color = (255, 0, 0)
    car = cv2.rectangle(img, (top_x, top_y), (bottom_x, bottom_y), bbox_color, bbox_width)
    car = cv2.circle(car, (center_x, center_x), radius=0, color=(0, 0, 255), thickness=-1)

    plt.figure(figsize=(20, 10))
    plt.imshow(car)
    plt.show()

    # cropping the number plate to display
    number_plate = Image.fromarray(img).crop((top_x + bbox_width, top_y + bbox_width, bottom_x, bottom_y))
    plt.figure(figsize=(5, 5))
    plt.imshow(number_plate)
    plt.show()
display_car_image(10)
#FUNCTION FOR PLOTTING
def plot_loss_acc(acc,val_acc,loss,val_loss):

    epochs=range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.figure()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.grid()
#Creating Training and Validation Data Generators
datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.1,
                            )

train_generator = datagen.flow_from_dataframe(
    df,
    directory="IndianPlates/",
    x_col="image_name",
    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],
    target_size=(WIDTH, HEIGHT),
    batch_size=32, 
    class_mode="other",
    subset="training")

validation_generator = datagen.flow_from_dataframe(
    df,
    directory="IndianPlates/",
    x_col="image_name",
    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],
    target_size=(WIDTH, HEIGHT),
    batch_size=32, 
    class_mode="other",
    subset="validation")
#Estimating Step size for train and validation set
STEP_SIZE_TRAIN = int(np.ceil(train_generator.n / train_generator.batch_size))
STEP_SIZE_VAL = int(np.ceil(validation_generator.n / validation_generator.batch_size))

print("Train step size:", STEP_SIZE_TRAIN)
print("Validation step size:", STEP_SIZE_VAL)

train_generator.reset()
validation_generator.reset()
#Creating CNN
cnnmodel = Sequential()

cnnmodel.add(Conv2D(64, (3,3), activation='relu', input_shape=(WIDTH,HEIGHT,3)))
cnnmodel.add(MaxPooling2D(2,2))
cnnmodel.add(Conv2D(32, (3,3), activation='relu'))
cnnmodel.add(MaxPooling2D(2,2))
cnnmodel.add(Conv2D(16, (3,3), activation='relu'))
cnnmodel.add(MaxPooling2D(2,2))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(128, activation="relu"))
cnnmodel.add(Dense(64, activation="relu"))
cnnmodel.add(Dense(64, activation="relu"))
cnnmodel.add(Dense(4, activation="sigmoid"))

cnnmodel.summary()
cnnmodel.compile(optimizer=Adam(lr=0.0005), loss="mse", metrics=['acc'])
history = cnnmodel.fit_generator(train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=30)
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

plot_loss_acc(acc,val_acc,loss,val_loss)
VGG16model = Sequential()

VGG16model.add(VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3)))
VGG16model.add(Flatten())
VGG16model.add(Dense(128, activation="relu"))
VGG16model.add(Dense(64, activation="relu"))
VGG16model.add(Dense(64, activation="relu"))
VGG16model.add(Dense(4, activation="sigmoid"))

VGG16model.layers[-6].trainable = False

VGG16model.summary()
VGG16model.compile(optimizer=Adam(lr=0.0005), loss="mse", metrics=['acc'])
history = VGG16model.fit_generator(train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=30)
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

plot_loss_acc(acc,val_acc,loss,val_loss)
MNV2model = Sequential()

MNV2model.add(MobileNetV2(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3)))
MNV2model.add(Flatten())
MNV2model.add(Dense(128, activation="relu"))
MNV2model.add(Dense(64, activation="relu"))
MNV2model.add(Dense(64, activation="relu"))
MNV2model.add(Dense(4, activation="sigmoid"))

MNV2model.layers[-6].trainable = False

MNV2model.summary()
MNV2model.compile(optimizer=Adam(lr=0.0005), loss="mse", metrics=['acc'])
history = MNV2model.fit_generator(train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=30)
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

plot_loss_acc(acc,val_acc,loss,val_loss)
#HEAD
Headmodel = Sequential()

Headmodel.add(Dense(128, activation="relu"))
Headmodel.add(Dense(64, activation="relu"))
Headmodel.add(Dense(64, activation="relu"))
Headmodel.add(Dense(4, activation="sigmoid"))

mergedOutput = Concatenate()([MNV2model.output, VGG16model.output])
finalOutput = Headmodel(mergedOutput) 
fullModel = Model([MNV2model.input,VGG16model.input],finalOutput)
fullModel.compile(optimizer=Adam(lr=0.0005), loss="mse", metrics=['acc'])
fullModel = MNV2model.fit_generator(train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=30)















