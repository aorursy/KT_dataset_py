# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PATH = ("/kaggle/input/just-helping-a-friend/Image_2/Input/Dataset")

#Test_Path = ("/kaggle/input/face-recognition/test/test")

Test_path = ("/kaggle/input/just-helping-a-friend/Image_2/Input/Dataset/test")
DATA_PATH = os.path.join(PATH, 'train')

#TEST_PATH = os.path.join(Test_path, 'New folder')

test_dir_list=os.listdir(Test_path)

data_dir_list = os.listdir(DATA_PATH)

print(data_dir_list)

print(test_dir_list)
img_rows=400

img_cols=400

num_channel=3



num_epoch = 15

batch_size = 32



img_data_list=[]

classes_names_list=[]

target_column=[]

img_data_list_test=[]
import cv2

for dataset in data_dir_list:

    classes_names_list.append(dataset)

    print("Getting images from {} folder\n".format(dataset))

    img_list = os.listdir(DATA_PATH +'/'+ dataset)

    for img in img_list:

        input_img = cv2.imread(DATA_PATH + '/' + dataset + '/' + img)

        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))

        img_data_list.append(input_img_resize)

        target_column.append(dataset)
test_data = []

test_list = os.listdir(Test_path)

for img in test_list:

        input_img = cv2.imread(Test_path + '/' + img)

        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))

        test_data.append(input_img_resize)
num_classes = len(classes_names_list)

print(num_classes)
#normalizing it 

img_data = np.array(img_data_list)

img_data = img_data.astype('float32')

img_data /= 255

test_data = np.array(test_data)

test_data = test_data.astype('float32')

test_data /= 255
#resizing the images and the num of samples

print(len(img_data))

print(img_data.shape)

print(len(test_data))

print(test_data.shape)
#what is happening?

classes_names_list
from sklearn.preprocessing import LabelEncoder

Labelencoder = LabelEncoder()

target_column = Labelencoder.fit_transform(target_column)
#from collections import Counter

#Counter(classes).values()

classes = target_column

    # classes = np.ones((num_of_samples,),dtype = 'int64')

classes
from keras.utils import to_categorical



classes = to_categorical(classes, num_classes)
classes.shape
from sklearn.utils import shuffle



X, Y = shuffle(img_data, classes, random_state=123)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

g = plt.imshow(X_train[3][:,:,0])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
num_of_samples = img_data.shape[0]

input_shape = img_data[0].shape

print(input_shape)
model = Sequential()



model.add(Conv2D(16,(3,3),activation = "relu", input_shape=input_shape))

model.add(Conv2D(16,(3,3),activation = "relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.6))



model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.6))

model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

model.summary()
cnn_1 = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
from pylab import rcParams

rcParams['figure.figsize'] = 10, 4

plt.plot(cnn_1.history['accuracy'])

plt.plot(cnn_1.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_1.history['loss'])

plt.plot(cnn_1.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
import tensorflow as tf
tf.keras.utils.plot_model(

    model, to_file='model.png', show_shapes=False, show_layer_names=True,

    rankdir='TB', expand_nested=False, dpi=96

)
#from sklearn.metrics import confusion_matrix

results = model.predict(test_data)

results.shape
print(model.predict_classes(test_data))
print(plt.imshow(test_data[1][:,:,0]))