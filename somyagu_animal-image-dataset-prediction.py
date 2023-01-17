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
#load image
data='/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/'
import os
from os import listdir
listdir("/kaggle/input")
listdir("/kaggle/input/animal-image-datasetdog-cat-and-panda")
listdir("/kaggle/input/animal-image-datasetdog-cat-and-panda/animals")
listdir("/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals")
listdir("/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals/cats")
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
x=[]
y=[]

k="/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals/"
for file1 in listdir(k):
    file2=k+"/"+file1
    for file3 in listdir(file2):
        file4=file2+"/"+file3
        image = load_img(file4,target_size=(108,108,3))
        img_array = img_to_array(image)
        x.append(img_array)
        y.append(file1)
len(x),len(y)
import numpy as np
x=np.array(x)
y=np.array(y)
x.shape,y.shape
from sklearn.preprocessing import LabelEncoder
k = LabelEncoder()
y= k.fit_transform(y)
from keras.utils import to_categorical
y=to_categorical(y)
x=x/255

y.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
from keras.applications.vgg16 import VGG16
model=VGG16(input_shape=(108,108,3),include_top=False)
from keras import Model
from keras.optimizers import SGD
from keras.layers import Flatten, Dense
def Animal_Image():
    model=VGG16(input_shape=(108,108,3),include_top=False)
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
    class2 = Dense(62, activation="relu", kernel_initializer="he_uniform")(class1)
    output = Dense(3, activation="softmax")(class1)
    model1 = Model(inputs=model.inputs, outputs=output)
    opt = SGD(lr=0.01, momentum=0.9)
    model1.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])
    return model1
obj=Animal_Image()
objective=obj.fit(x_train,y_train,batch_size=32, validation_data=(x_test,y_test),epochs=12,verbose=1)
_, acc =obj.evaluate(x_test,y_test, verbose=0)
acc
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.layers import Dropout
# define cnn model
def define_model():
    
    model = ResNet50(include_top=False, input_shape=(108, 108, 3))

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
    class2 = Dense(62, activation="relu", kernel_initializer="he_uniform")(class1)
    output = Dense(3, activation="softmax")(class1)
# define new model
    model = Model(inputs=model.inputs, outputs=output)

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
k2=define_model()
k=k2.fit(x_train,y_train,batch_size=32, validation_data=(x_test,y_test),epochs=12,verbose=1)
_, acc =k2.evaluate(x_test, y_test, verbose=0)
acc
