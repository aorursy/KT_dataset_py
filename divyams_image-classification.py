import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "/kaggle/input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import os

os.getcwd()
import tensorflow as tf 
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
import warnings

warnings.filterwarnings('ignore')



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from keras.utils import to_categorical

from PIL import Image 

from keras.models  import Model,Sequential

from keras.layers import Flatten,Dense, Dropout

from keras.layers import Convolution2D, MaxPool2D

from keras.layers import BatchNormalization, GlobalAveragePooling2D

from keras.optimizers import Adam 

from keras.applications.inception_v3 import InceptionV3
import pandas as pd

import os

import cv2
os.getcwd()
PATH = ("/kaggle/input/face-recognition/data")

Test_Path = ("/kaggle/input/face-recognition/test/test")
DATA_PATH = os.path.join(PATH, 'Data')

data_dir_list = os.listdir(DATA_PATH)

print(data_dir_list)
img_rows=224

img_cols=224

num_channel=3



num_epoch = 25

batch_size = 32



img_data_list=[]

classes_names_list=[]

target_column=[]
for dataset in data_dir_list:

    classes_names_list.append(dataset)

    print("Getting images from {} folder\n".format(dataset))

    img_list = os.listdir(DATA_PATH +'/'+ dataset)

    for img in img_list:

        input_img = cv2.imread(DATA_PATH + '/' + dataset + '/' + img)

        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))

        img_data_list.append(input_img_resize)

        target_column.append(dataset)
target_column
num_classes = len(classes_names_list)

print(num_classes)
#normalizing it 

img_data = np.array(img_data_list)

img_data = img_data.astype('float32')

img_data /= 255
#resizing the images and the num of samples

print(len(img_data))

print(img_data.shape)
#what is happening?

classes_names_list
num_of_samples = img_data.shape[0]

input_shape = img_data[0].shape
print(input_shape)
from sklearn.preprocessing import LabelEncoder

Labelencoder = LabelEncoder()

target_column = Labelencoder.fit_transform(target_column)
from collections import Counter

#Counter(classes).values()
classes = target_column

    # classes = np.ones((num_of_samples,),dtype = 'int64')
classes
classes.shape
from keras.utils import to_categorical



classes = to_categorical(classes, num_classes)
from sklearn.utils import shuffle



X, Y = shuffle(img_data, classes, random_state=123)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
y_test.shape
y_test
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

model.summary()
cnnfr = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
DATA_PATH_test = Test_Path

data_dir_list = os.listdir(DATA_PATH_test)

print(data_dir_list)
test_Y = []

img_test_data_list = []
for dataset in data_dir_list:

   # classes_names_list.append(dataset)

    print("Getting images from {} folder\n".format(dataset))

    img_list_test = os.listdir(DATA_PATH_test+'/'+ dataset)

    for img in img_list_test:

        input_img_test = cv2.imread(DATA_PATH_test + '/' + dataset + '/' + img)

        input_img_test_resize=cv2.resize(input_img_test,(img_rows,img_cols))

        img_test_data_list.append(input_img_test_resize)

        test_Y.append(dataset)
test_Y
#normalizing it 

img_test_data = np.array(img_test_data_list)

img_test_data = img_data.astype('float32')

img_test_data /= 255
img_test_data_list
from sklearn.preprocessing import LabelEncoder

Labelencoder = LabelEncoder()

test_Y = Labelencoder.fit_transform(test_Y)
test_Y[5:10]
#

#Y = shuffle(img_data, classes, random_state=123)

X_test, y_test = shuffle(img_test_data, classes, random_state=123)
score = model.evaluate(X_test, y_test, batch_size=32)



print('Test Loss:', score[0])

print('Test Accuracy:', score[1])