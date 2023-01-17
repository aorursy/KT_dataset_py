# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
folder="/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals/"
import os
from os import listdir
listdir(folder)
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
listdir("/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals/")
listdir("/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals/cats")
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
import numpy
from numpy import asarray
x=asarray(x)
y=asarray(y)
x.shape
y.shape
y
from sklearn.preprocessing import LabelEncoder
m= LabelEncoder()
y = m.fit_transform(y)

y

from keras.utils import to_categorical
y = to_categorical(y)
y.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/2, random_state = 0)
import numpy as np
np.unique(y)
x_train=x_train/255
x_test=x_test/255
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
# example of loading the vgg16 model
from keras.applications.vgg16 import VGG16
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(300, 300, 3))
## example of loading the resnet50 model
#from keras.applications.resnet50 import ResNet50


from keras import Model
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
k4=define_model()
k4.summary()
history=k4.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=5, batch_size=32,verbose=1)
from matplotlib import pyplot
pyplot.title("Classification Accuracy")
pyplot.plot(history.history["accuracy"], color="blue", label="train")
pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
