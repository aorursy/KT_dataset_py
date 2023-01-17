import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import os,cv2

from tqdm import tqdm

import keras

from sklearn.model_selection import train_test_split



import keras

from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from pathlib import Path

from keras import backend as K





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



#TestRun

img=mpimg.imread('../input/stanford-dogs-dataset/images/Images/n02085620-Chihuahua/n02085620_10074.jpg')

imgplot = plt.imshow(img)

plt.show()





#Functions 

def genarateLabel(folderName):

    return (folderName.split("-")[1])



def resize_with_padding(img,size):



  # calculate the ratio

  old_size = img.shape[:2] 

  ratio = float(size)/max(old_size)

  new_size = tuple([int(x*ratio) for x in old_size])

  

  # resize the image to the appropriate size.

  img = cv2.resize(img,(new_size[1], new_size[0]))

  

  delta_w = size - new_size[1]

  delta_h = size - new_size[0]

  top, bottom = delta_h//2, delta_h-(delta_h//2)

  left, right = delta_w//2, delta_w-(delta_w//2)

  

  color = [0, 0, 0]

  new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,

    value=color)

  return new_im





# Define variables

All_Data = []

All_Labels = []

Classes = {}

imgsize = 32

classID = 0

imageFolderDir = '../input/stanford-dogs-dataset/images/Images/'



# Genarate dataset 

for foldername in tqdm(os.listdir(imageFolderDir)):

  #genarate class name from folder.

  label = genarateLabel(foldername)

  Classes[classID] =label



  #get path for each folder.

  folderpath = os.path.join('../input/stanford-dogs-dataset/images/Images/',foldername)



  #loop via each folder.

  for image in os.listdir(folderpath):



      #get image path

      imagepath = os.path.join(folderpath,image)



      #get image and resize

      img = cv2.imread(imagepath,cv2.IMREAD_COLOR)

      img = resize_with_padding(img,imgsize)

      #img = cv2.resize(img,(imgsize,imgsize))



      #append to all_data file

      All_Data.append(np.array(img))

      All_Labels.append([classID])



  classID = classID +1

 

All_Data = np.array(All_Data)

All_Labels = np.array(All_Labels)

All_Data = np.array(All_Data)

All_Labels = keras.utils.to_categorical(All_Labels, 120)

All_Data = All_Data/255;

x_train,x_test,y_train,y_test = train_test_split(All_Data,All_Labels,test_size=0.3,random_state=69)


K.tensorflow_backend._get_available_gpus()





# Create a model and add layers

model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32,32,3), activation="relu"))

model.add(Conv2D(32, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(120, activation="softmax"))



# Compile the model

model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



# Train the model

model.fit(

    x_train,

    y_train,

    batch_size=64,

    epochs=120,

    validation_data=(x_test, y_test),

    shuffle=True

)



# Save neural network structure

model_structure = model.to_json()

f = Path("model_structure.json")

f.write_text(model_structure)



# Save neural network's trained weights

model.save_weights("model_weights.h5")