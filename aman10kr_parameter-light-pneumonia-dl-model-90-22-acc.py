
import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
train_data = []
gamma = 2.5
normal_count = 0
pneumonia_count = 0
img_size = 124
assign_dict = {"NORMAL":0, "PNEUMONIA":1}
directory = "../input/chest-xray-pneumonia/chest_xray/train"
for sub_directory in os.listdir(directory):
    if sub_directory == "NORMAL":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            normal_count += 1
            img = cv2.imread(os.path.join(inner_directory,i),0)
            img =  adjust_gamma(img, gamma=gamma)
            img = cv2.resize(img,(img_size,img_size))
            train_data.append([img,assign_dict[sub_directory]])
    if sub_directory == "PNEUMONIA":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            pneumonia_count +=1
            img = cv2.imread(os.path.join(inner_directory,i),0)
            img =  adjust_gamma(img, gamma=gamma)
            img = cv2.resize(img,(img_size,img_size))
            train_data.append([img,assign_dict[sub_directory]])
random.shuffle(train_data)
        
print(normal_count,pneumonia_count)
val_data = []
directory = "../input/chest-xray-pneumonia/chest_xray/val"
for sub_directory in os.listdir(directory):
    if sub_directory == "NORMAL":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            img = cv2.imread(os.path.join(inner_directory,i),0)
            img =  adjust_gamma(img, gamma=gamma)
            img = cv2.resize(img,(img_size,img_size))
            val_data.append([img,assign_dict[sub_directory]])
    if sub_directory == "PNEUMONIA":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            img = cv2.imread(os.path.join(inner_directory,i),0)
            img =  adjust_gamma(img, gamma=gamma)
            img = cv2.resize(img,(img_size,img_size))
            val_data.append([img,assign_dict[sub_directory]])
random.shuffle(val_data)
test_data = []
directory = "../input/chest-xray-pneumonia/chest_xray/test"
for sub_directory in os.listdir(directory):
    if sub_directory == "NORMAL":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            img = cv2.imread(os.path.join(inner_directory,i),0)
            img =  adjust_gamma(img, gamma=gamma)
            img = cv2.resize(img,(img_size,img_size))
            test_data.append([img,assign_dict[sub_directory]])
    if sub_directory == "PNEUMONIA":
        inner_directory = os.path.join(directory,sub_directory)
        for i in os.listdir(inner_directory):
            img = cv2.imread(os.path.join(inner_directory,i),0)
            img =  adjust_gamma(img, gamma=gamma)
            img = cv2.resize(img,(img_size,img_size))
            test_data.append([img,assign_dict[sub_directory]])
random.shuffle(test_data)
train_X = []
train_Y = []
for features,label in train_data:
    train_X.append(features)
    train_Y.append(label)
val_X = []
val_Y = []
for features,label in val_data:
    val_X.append(features)
    val_Y.append(label)
test_X = []
test_Y = []
for features,label in test_data:
    test_X.append(features)
    test_Y.append(label)
train_X = np.array(train_X)/255.0
train_X = train_X.reshape(-1,124,124,1)
train_Y = np.array(train_Y)
val_X = np.array(val_X)/255.0
val_X = val_X.reshape(-1,124,124,1)
val_Y = np.array(val_Y)
test_X = np.array(test_X)/255.0
test_X = test_X.reshape(-1,124,124,1)
test_Y = np.array(test_Y)
normal = []
infected = []
fig=plt.figure(figsize=(12,12))
for i,cl in enumerate(train_Y):
    if cl == 0:
        if len(normal) == 2:
            continue
        normal.append(train_X[i])
    if cl== 1:
        if len(infected) == 2:
            continue
        infected.append(train_X[i])

fig.suptitle("Normal vs Pneumonia X-ray Images", fontsize=16)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.set_title("Normal")
ax1.imshow(np.squeeze(normal[0]))
ax2.set_title("Normal")
ax2.imshow(np.squeeze(normal[1]))
ax3.set_title("Pneumonia")
ax3.imshow(np.squeeze(infected[0]))
ax4.set_title("Pneumonia")
ax4.imshow(np.squeeze(infected[1]))
plt.show()
model = Sequential()

model.add(Conv2D(64, (3, 3), padding = "same", activation='relu',input_shape=(124,124,1)))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), padding = "same",activation='relu'))
model.add(Conv2D(32, (1, 1),padding = "same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, (3, 3), padding = "same",activation='relu'))
model.add(Conv2D(16, (1, 1), padding = "same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(8, (3, 3), padding = "same",activation='relu'))
model.add(Conv2D(8, (1, 1), padding = "same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer="adam",metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,    
        rotation_range=20,    
        width_shift_range=0.1,
        height_shift_range=0.1,  
        horizontal_flip=False,  
        vertical_flip=False)
datagen.fit(train_X)
from sklearn.model_selection import train_test_split
train_X, val_X_new, train_Y, val_Y_new = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)
val_X = np.vstack((val_X,val_X_new))
val_Y = np.concatenate([val_Y,val_Y_new])
weights = {0:5, 1:1}
history = model.fit(datagen.flow(train_X,train_Y, batch_size = 32) ,epochs = 100 ,class_weight=weights, validation_data = datagen.flow(val_X, val_Y))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
test_X.shape
score = model.evaluate(test_X, test_Y, verbose=0)
print("Loss: " + str(score[0]))
print("Accuracy: " + str(score[1]*100) + "%")
