# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

files=[]

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        files.append(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(files)

print(os.getcwd())
import tarfile

tar = tarfile.open('/kaggle/input/english-typed-alphabets-and-numbers-dataset/EnglishFnt.tgz')

tar.extractall()

tar.close()
PATH = os.getcwd()

PATH
data_path = os.path.join(PATH, './English/Fnt')

data_dir_list = os.listdir(data_path)

print(data_dir_list)
len(data_dir_list)
data_path

#os.listdir(data_path)
import cv2



#Read the images and store them in the list

class_count_start = 0



class_label = 0

classes = []

img_data_list=[]

classes_names_list=[]

img_rows=100

img_cols=100





old_name = data_dir_list[0]



for dataset in data_dir_list:

    classes_names_list.append(dataset) 

    print ('Loading images from {} folder\n'.format(dataset))



    img_list=os.listdir(data_path+'/'+ dataset)

    #print(len(img_list))

    if(old_name != dataset):

        class_count_start = class_count_start + 1

        old_name = dataset

    for img in img_list:

        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )

        #input_img=input_img/255

        #input_img1 = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        input_img_resize=cv2.resize(input_img, (img_rows, img_cols))

        img_data_list.append(input_img_resize)

        classes.append(class_label)

        #print(class_count_start,img)

        class_count_start = class_count_start + 1

    class_count_start = class_count_start - 1

    class_label = class_label + 1
1016*62
classes_names_list
len(classes)
from collections import Counter

len(Counter(classes).keys())
classes[62991]
class_labels = list(range(0,62,1))

#class_labels
map_labels = dict(zip(class_labels,classes_names_list))

print(map_labels)
import matplotlib.pyplot as plt

import numpy

print(classes[0])

print(map_labels[classes[0]])

plt.figure(figsize=(20, 4))

plt.imshow(img_data_list[0])
img = list(range(0,62992,1016))

img1 = list(range(1015,62992,1016))

img2 = list(range(253,62992,1016))

img3 = list(range(254,62992,1016))

img = img + img1 + img2 + img3

img.sort()

len(img)
img[0:4]
img[4:8]
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20, 45))



for i,j in enumerate(img):

    if i == 248:

        break

    plt.subplot(21,12,i+1)

    plt.imshow(img_data_list[j])

    plt.title(map_labels[classes[j]])
img_data_list[0].shape
#Convert class labels to numberic using on-hot encoding

from keras.utils import to_categorical



classes = to_categorical(classes, 62)

classes
from sklearn.model_selection import train_test_split



xtrain=[]

ytrain = []

xtest=[]

ytest = []



xtrainlist,  xtest_list, ytrainlist, ytest_list = train_test_split(img_data_list, classes, random_state=1234, test_size=0.3,shuffle = True)

xtrain.extend(xtrainlist)

ytrain.extend(ytrainlist)

xtest.extend(xtest_list)

ytest.extend(ytest_list)
del xtrainlist, xtest_list, ytrainlist, ytest_list
print(len(xtrain))

print(len(ytrain))

print(len(xtest))

print(len(ytest))
num_of_samples = len(xtrain)

input_shape = xtrain[0].shape

print(num_of_samples)

print(input_shape)
xtrain = np.array(xtrain)

xtest = np.array(xtest)
from keras.models import Sequential

from keras.layers import Activation,Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import initializers
num_classes=62



datagen = ImageDataGenerator(rescale=1.0/255.0)

train_iterator = datagen.flow(xtrain, ytrain, batch_size=32)

test_iterator = datagen.flow(xtest, ytest, batch_size=32)





model = Sequential()



model.add(Conv2D(32, (3, 3),kernel_initializer='random_normal', activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3),kernel_initializer='random_normal', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer='random_normal', activation='relu'))

model.add(Conv2D(32, (3, 3),kernel_initializer='random_normal', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(96, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
#compile the model

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=["accuracy"])
model.summary()
#hist_val = model.fit(xtrain, ytrain, batch_size=16, epochs=10, verbose=1, validation_split=0.20)
history = model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=30, validation_data=test_iterator)