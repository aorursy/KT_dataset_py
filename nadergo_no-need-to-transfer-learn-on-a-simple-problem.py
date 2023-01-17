import numpy as np
import statistics
import pandas as pd
# import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
# from os import walk
import os, glob
def create_data(dir):
    pixels = []
    hemmor_labels = []

    height  = 64
    width = 64
    for infile in glob.glob("/kaggle/input/brain-tumor-images-dataset/Brain Tumor Images Dataset/"+dir+"/hemmorhage_data/*.png"):
        img =  Image.open(infile)
        img = img.resize((width, height), Image.ANTIALIAS).convert("RGB")
        img = np.asarray(img)/255.0
        img = img.tolist()
        pixels.append(img)
        hemmor_labels.append(1)

    for infile in glob.glob("/kaggle/input/brain-tumor-images-dataset/Brain Tumor Images Dataset/"+dir+"/non_hemmorhage_data/*.png"):
        img =  Image.open(infile)
        img = img.resize((width, height), Image.ANTIALIAS).convert("RGB")
        img = np.asarray(img)/255.0
        img = img.tolist()
        pixels.append(img)
        hemmor_labels.append(0)

    data = {"pixels" : pixels, "hemmorhage" :hemmor_labels }

    data = pd.DataFrame(data)
    data = data.sample(frac=1).reset_index(drop=True)
    return data
train = create_data("training_set")
validation = create_data("valdation_set")
train= pd.concat([train,validation],axis=0)
xtrain = train["pixels"]
xtrain = np.asarray(xtrain.to_numpy().tolist())
ytrain = train[["hemmorhage"]]
ytrain = ytrain.to_numpy()
test = create_data("test_set")
xtest = test["pixels"]
xtest = np.asarray(xtest.to_numpy().tolist())
ytest = test[["hemmorhage"]]
ytest = ytest.to_numpy()
model = Sequential()

model.add(Conv2D(32,kernel_size= 3,padding='valid',activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size= 3,padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size= 3,padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=30)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show() # which shows that the model is slightly overfitting
img =  Image.open("/kaggle/input/brain-tumor-images-dataset/Brain Tumor Images Dataset/test_set/hemmorhage_data/003.png")
imgshow= np.asarray(img.resize((224, 224), Image.ANTIALIAS).convert("RGB"))
img= np.asarray(img.resize((64, 64), Image.ANTIALIAS).convert("RGB"))

imgn = img/255.0
imgnlist = [imgn.tolist()]
# print(imgnlist)
yp = model.predict(np.asarray(imgnlist))
print(yp)
title = "hemmorhage" if yp[0] >= 0.5 else "nonhemmorhage"
plt.title(title)
plt.imshow(np.asarray(imgshow))