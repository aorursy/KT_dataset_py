import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from glob import glob

from PIL import Image

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import load_img,img_to_array

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense ,Dropout,Flatten , Conv2D, MaxPooling2D

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
img=load_img("/kaggle/input/100-bird-species/180/train/ALBATROSS/028.jpg")

plt.imshow(img)

plt.axis("off")

plt.show()
shape_of_images=img_to_array(img)

print("Shape of Images =",shape_of_images.shape)
def read_images(path,num_img):

    array=np.zeros((num_img,224,224,3))

    i=0

    for img in os.listdir(path):

        img_path=path+"/"+img

        img=Image.open(img_path,mode="r")

        data=np.asarray(img,dtype="uint8")

        array[i]=data

        i+=1

    return array

#                                              TRAIN SECTION

train_albatros_path=r"/kaggle/input/100-bird-species/180/train/ALBATROSS"

num_train_albatros=len(glob("/kaggle/input/100-bird-species/180/train/ALBATROSS/*"))

train_albatros_array=read_images(train_albatros_path,num_train_albatros)



train_bald_eagle_path=r"/kaggle/input/100-bird-species/180/train/BALD EAGLE"

num_train_bald_eagle=len(glob("/kaggle/input/100-bird-species/180/train/BALD EAGLE/*"))

train_bald_eagle_array=read_images(train_albatros_path,num_train_bald_eagle)



train_american_goldfinch_path=r"/kaggle/input/100-bird-species/180/train/AMERICAN GOLDFINCH"

num_train_american_goldfinch=len(glob("/kaggle/input/100-bird-species/180/train/AMERICAN GOLDFINCH/*"))

train_american_goldfinch_array=read_images(train_american_goldfinch_path,num_train_american_goldfinch)

#                                              VALIDATION SECTION                          

validation_albatros_path=r"/kaggle/input/100-bird-species/180/valid/ALBATROSS"

num_validation_albatros=len(glob("/kaggle/input/100-bird-species/180/valid/ALBATROSS/*"))

validation_albatros_array=read_images(validation_albatros_path,num_validation_albatros)



validation_bald_eagle_path=r"/kaggle/input/100-bird-species/180/valid/BALD EAGLE"

num_validation_bald_eagle=len(glob("/kaggle/input/100-bird-species/180/valid/BALD EAGLE/*"))

validation_bald_eagle_array=read_images(validation_bald_eagle_path,num_validation_bald_eagle)



validation_american_goldfinch_path=r"/kaggle/input/100-bird-species/180/valid/AMERICAN GOLDFINCH"

num_validation_american_goldfinch=len(glob("/kaggle/input/100-bird-species/180/valid/AMERICAN GOLDFINCH/*"))

validation_american_goldfinch_array=read_images(validation_american_goldfinch_path,num_validation_american_goldfinch)
print("train albatros shape =",train_albatros_array.shape)

print("train bald eagle shape =",train_bald_eagle_array.shape)

print("train american goldfinch shape =",train_american_goldfinch_array.shape)

print("validation albatros shape =",validation_albatros_array.shape)

print("validation bald eagle shape =",validation_bald_eagle_array.shape)

print("validation american goldfinch shape =",validation_american_goldfinch_array.shape)
img=train_albatros_array[48]

plt.imshow(img.astype(np.uint8))

plt.axis("off")

plt.show()
img1=train_american_goldfinch_array[65]

plt.imshow(img1.astype(np.uint8))

plt.axis("off")

plt.show()
img2=train_bald_eagle_array[36]

plt.imshow(img2.astype(np.uint8))

plt.axis("off")

plt.show()
x_train=np.concatenate((train_albatros_array,train_bald_eagle_array,train_american_goldfinch_array),axis=0)

x_val=np.concatenate((validation_albatros_array,validation_bald_eagle_array,validation_american_goldfinch_array),axis=0)

print("x_train shape =",x_train.shape)

print("x_val shape =",x_val.shape)
def resize_images(img):

    array=np.zeros((img.shape[0],100,100,3))

    for i in range(img.shape[0]):

        array[i] = cv2.resize(img[i,:,:,:],(100,100))

    return array

x_train=resize_images(x_train)

x_val=resize_images(x_val)
print("new x train shape :",x_train.shape)

print("new x validation shape :",x_val.shape)
"""

albatros = 0

bald eagle = 1

american goldfinch = 2

"""

y_train=np.concatenate((np.zeros(100),np.ones(162),np.full(134,2)),axis=0)

y_val=np.concatenate((np.zeros(5),np.ones(5),np.full(5,2)),axis=0)

print("y_train shape =",y_train.shape)

print("y_val shape =",y_val.shape)
y_train = to_categorical(y_train,3)

y_val = to_categorical(y_val,3)

print("y_train shape =",y_train.shape)

print("y_val shape =",y_val.shape)
model=Sequential()

model.add(Conv2D(20,(3,3),activation="relu",padding="same",input_shape=(100,100,3)))

model.add(MaxPooling2D())



model.add(Conv2D(20,(3,3),activation="relu",padding="same"))

model.add(MaxPooling2D())



model.add(Conv2D(20,(3,3),activation="relu",padding="same"))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dense(206,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(103,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(3,activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
x_train=x_train / 255.0

x_val=x_val / 255.0
history = model.fit(x_train,y_train,batch_size=250, epochs = 100, validation_data = (x_val,y_val))
print(history.history.keys())
plt.plot(history.history["loss"],color="red",label="Train Loss")

plt.plot(history.history["val_loss"],color="green",label="Validation Loss")

plt.legend()

plt.title("Loss Plot")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss Values")

plt.grid()

plt.show()
plt.plot(history.history["accuracy"],color="red",label="Train Accuracy")

plt.plot(history.history["val_accuracy"],color="green",label="Validation Accuracy")

plt.legend()

plt.title("Accuracy Plot")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy Values")

plt.grid()

plt.show()
prediction=model.predict(x_val)
predicted_classes=np.argmax(prediction,axis=1)

y_true=np.argmax(y_val,axis=1)

print("predicted classes shape :",predicted_classes.shape)

print("y true shape :",y_true.shape)
cfm=confusion_matrix(y_true,predicted_classes)

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(cfm,annot=True,cmap="coolwarm",linecolor="black",linewidths=1,fmt=".0f",ax=ax)

plt.xlabel("Real Labels")

plt.ylabel("Predicted Labels")

plt.show()

"""

albatros = 0

bald eagle = 1

american goldfinch = 2

"""