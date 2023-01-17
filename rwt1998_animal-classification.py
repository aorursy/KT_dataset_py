import os
from os import listdir
listdir("../input/animal-image-datasetdog-cat-and-panda/animals/animals")
folder = "../input/animal-image-datasetdog-cat-and-panda/animals/animals"
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
folder1="/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals/"
x=[]
y=[]
for file1 in listdir(folder):
    file2=folder1+"/"+file1
    for file3 in listdir(file2):
        file4=file2+"/"+file3
        image = load_img(file4,target_size=(300,300))
        image=img_to_array(image)
        x.append(image)
        y.append(file1)
len(x)
len(y)
import numpy as np
from numpy import asarray
x=asarray(x)
y=asarray(y)
x.shape
y.shape
from sklearn.preprocessing import LabelEncoder
m= LabelEncoder()
y = m.fit_transform(y)
y
from keras.utils import to_categorical
y = to_categorical(y)
y.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/2, random_state = 0)
np.unique(y)
x_train=x_train/255
x_test=x_test/255
x_train.shape
y_train.shape
x_test.shape
y_test.shape
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras import Model
from keras.applications.vgg16 import VGG16
model = VGG16(include_top=False, input_shape=(300, 300, 3))
# define cnn model
def define_model():
# load model
  model = VGG16(include_top=False, input_shape=(300, 300, 3))
# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
# add new classifier layers
  flat1 = Flatten()(model.layers[-1].output)
  class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
 
  output = Dense(3, activation="softmax")(class1)
# define new model
  model = Model(inputs=model.inputs, outputs=output)
# compile model
  opt = Adam(lr=0.01)
  model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
  return model
model_2=define_model()
model_2.summary()
model_2.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=3, batch_size=32,verbose=1)