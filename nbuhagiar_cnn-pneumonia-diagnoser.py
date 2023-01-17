import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import os
print(os.listdir("../input/chest_xray/chest_xray/"))
print(os.listdir("../input/chest_xray/chest_xray/train/"))
file_loc = "../input/chest_xray/chest_xray/"
train_n = os.listdir(file_loc + "train/NORMAL/")
train_p = os.listdir(file_loc + "train/PNEUMONIA/")
fig, axarr = plt.subplots(3, 2, figsize=(16, 16))
axarr[0][0].set_title("Normal Sample Cases")
axarr[0][1].set_title("Pneumonia Sample Cases")
for i in range(3):
    axarr[i][0].imshow(cv2.imread(file_loc + "train/NORMAL/" + train_n[i]))
    axarr[i][0].axis("off")
    axarr[i][1].imshow(cv2.imread(file_loc + "train/PNEUMONIA/" + train_p[i]))
    axarr[i][1].axis("off")
sns.barplot(x=["Normal", "Pneumonia"], y=[len(train_n), len(train_p)])
train_images = train_n + train_p
train_images = [img for img in train_images if img != ".DS_Store"]
val_images = os.listdir(file_loc + "val/NORMAL/") + os.listdir(file_loc + "val/PNEUMONIA/")
val_images = [img for img in val_images if img != ".DS_Store"]
test_images = os.listdir(file_loc + "test/NORMAL/") + os.listdir(file_loc + "test/PNEUMONIA/")
test_images = [img for img in test_images if img != ".DS_Store"]

sns.barplot(x=["Train", "Validation", "Test"], y=[len(train_images), len(val_images), len(test_images)])
print("There are {} images in the training set.".format(len(train_images)))
print("There are {} images in the validation set.".format(len(val_images)))
print("There are {} images in the test set.".format(len(test_images)))
image_height, image_width = 256, 256
batch_size=32

data_generator_train = ImageDataGenerator(rescale=1/255, 
                                          width_shift_range=0.1, 
                                          height_shift_range=0.1, 
                                          zoom_range=0.1)
train = data_generator_train.flow_from_directory(directory=file_loc + "train/", 
                                                 target_size=(image_height, 
                                                              image_width), 
                                                 class_mode="binary", 
                                                 batch_size=batch_size)

data_generator_val = ImageDataGenerator(rescale=1/255)
val = data_generator_val.flow_from_directory(directory=file_loc + "val/", 
                                             target_size=(image_height, 
                                                          image_width), 
                                             class_mode="binary", 
                                             batch_size=batch_size)

data_generator_test = ImageDataGenerator(rescale=1/255)
test = data_generator_test.flow_from_directory(directory=file_loc + "test/", 
                                               target_size=(image_height, 
                                                            image_width), 
                                               class_mode="binary", 
                                               batch_size=batch_size)
model = Sequential()
model.add(Conv2D(64, 
                 (3, 3), 
                 input_shape=(image_height, image_width, 3), 
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
num_epochs = 10
history = model.fit_generator(train, 
                              steps_per_epoch=5216//batch_size, 
                              epochs=num_epochs, 
                              validation_data=val, 
                              validation_steps=16, 
                              callbacks=[ReduceLROnPlateau(patience=2, verbose=1)])
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
axarr[0].set_xlabel("Number of Epochs")
axarr[0].set_ylabel("Loss")
sns.lineplot(x=range(1, num_epochs+1), y=history.history["loss"], label="Train", ax=axarr[0])
sns.lineplot(x=range(1, num_epochs+1), y=history.history["val_loss"], label="Validation", ax=axarr[0])
axarr[1].set_xlabel("Number of Epochs")
axarr[1].set_ylabel("Accuracy")
axarr[1].set_ylim(0, 1)
sns.lineplot(x=range(1, num_epochs+1), y=history.history["acc"], label="Train", ax=axarr[1])
sns.lineplot(x=range(1, num_epochs+1), y=history.history["val_acc"], label="Validation", ax=axarr[1])
test_results = model.evaluate_generator(test, steps=624//batch_size)
print("The model has a test accuracy of {}.".format(test_results[1]))
