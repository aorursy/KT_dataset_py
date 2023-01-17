# importing data from kaggle so we first install kaggle
! pip install -q kaggle
from google.colab import files
# upload the Kaggle.json API Key file from your account to link with Kaggle
files.upload()
 ! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json 
#changing the permission from kaggle to use dataset 
! kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# upload dataset API command 
ls
! unzip chest-xray-pneumonia.zip
ls
test_Pneumonia_dir = "./chest_xray/test/PNEUMONIA"
train_Pneumonia_dir = "./chest_xray/train/PNEUMONIA"
validation_Pneumonia_dir = "./chest_xray/val/PNEUMONIA"

test_Normal_dir = "./chest_xray/test/NORMAL"
train_Normal_dir = "./chest_xray/train/NORMAL"
validation_Normal_dir = "./chest_xray/val/NORMAL"
import os
print(len(os.listdir(train_Pneumonia_dir)))
print(len(os.listdir(train_Normal_dir)))
Pneumonia_images = os.listdir(train_Pneumonia_dir)
Pneumonia_images
from PIL import Image
Image.open(train_Pneumonia_dir +"/person1142_virus_1892.jpeg") #virus 
Image.open(train_Pneumonia_dir +"/person614_bacteria_2483.jpeg") #bacteria
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = "./chest_xray/train"
data_gen = ImageDataGenerator(1/255.0)
train_generator = data_gen.flow_from_directory(train_dir, target_size=(150, 150), class_mode="categorical" )
# checking Indecies of categorical classifcation in the train data
train_generator.class_indices
# building the model

model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", input_shape = (150, 150, 3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(512, activation="relu"))
#output layer
model.add(Dense(2, activation="softmax"))
model.compile(loss = "categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

H = model.fit(train_generator, epochs=20, verbose=1)
# Accuracy of max 97 percent is seen from the trained model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from google.colab import files
files.upload()
# we upload a pneumonia detected x-ray from test-data set to check the prediction 

image = load_img("person103_bacteria_489.jpeg", target_size=(150, 150))
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)
model.predict_classes(image)
# array of 1 is PNEUMONIA and can be checked below in class_indices
train_generator.class_indices
