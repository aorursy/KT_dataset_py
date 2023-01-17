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
import numpy as np

import pandas as pd

import os

import tarfile

import keras
def png_files(members):

    print(members)

    for tarinfo in members:

        if os.path.splitext(tarinfo.name)[1] == ".png":

            yield tarinfo



tar = tarfile.open("/kaggle/input/english-typed-alphabets-and-numbers-dataset/EnglishFnt.tgz")

#tar1 = tarfile.open("/kaggle/input/english-typed-alphabets-and-numbers-dataset/EnglishImg.tgz")

tar.extractall('./EnglishFnt',members=png_files(tar))

#tar1.extractall('./EnglishImg',members=png_files(tar1))

#print(tar.getnames())

tar.close()
os.listdir()
PATH = os.getcwd()

PATH
data_path = os.path.join(PATH, './EnglishFnt/English/Fnt')

data_dir_list = os.listdir(data_path)

print(data_dir_list)
import cv2



#Read the images and store them in the list

class_count_start = 0



class_label = 0

classes = []

img_data_list=[]

classes_names_list=[]

img_rows=50

img_cols=50





old_name = data_dir_list[0]



for dataset in data_dir_list:

    classes_names_list.append(dataset) 

    print ('Loading images from {} folder\n'.format(dataset)) 

    img_list=os.listdir(data_path+'/'+ dataset)

    if(old_name != dataset):

        class_count_start = class_count_start + 1

        old_name = dataset

    for img in img_list:

        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )

        #input_img1 = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        input_img_resize=cv2.resize(input_img, (img_rows, img_cols))



        img_data_list.append(input_img_resize)

        classes.append(class_label)

        #print(class_count_start,img)

        class_count_start = class_count_start + 1

    class_count_start = class_count_start - 1

    class_label = class_label + 1

    
import matplotlib.pyplot as plt

import numpy



plt.figure(figsize=(20, 4))

plt.imshow(img_data_list[8545])
from collections import Counter

len(Counter(classes).keys())

class_labels = list(range(0,62,1))
map_labels = dict(zip(class_labels,classes_names_list))

print(map_labels)
#Convert class labels to numberic using on-hot encoding

from keras.utils import to_categorical



classes = to_categorical(classes, 62)

classes
from sklearn.model_selection import train_test_split



xtrain=[]

ytrain = []

xtest=[]

ytest = []



xtrainlist,  xtest_list, ytrainlist, ytest_list = train_test_split(img_data_list, classes, random_state=1234, test_size=0.02,shuffle = True)

xtrain.extend(xtrainlist)

ytrain.extend(ytrainlist)

xtest.extend(xtest_list)

ytest.extend(ytest_list)

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D,BatchNormalization

from keras.optimizers import Adam
num_classes=62

model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50,50,3)))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(32, (3, 3), activation='tanh'))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='tanh'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
#complie the model

opt =Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.summary()
#training and fitting the model

hist = model.fit(np.array(xtrain), np.array(ytrain), epochs=15, verbose=1, batch_size =34,validation_split = 0.20)
#plot the evaluation of loss and accuracy on the train and validation sets

import matplotlib.pyplot as plt

plt.figure(figsize =(10,5))

plt.subplot(1,2,1)

plt.suptitle('Optimizer : Adam with learning rate 0.001', fontsize=10)

plt.ylabel('Loss', fontsize = 16)

plt.plot(hist.history['loss'], label = 'Training Loss')

plt.plot(hist.history['val_loss'], label = 'Validation Loss')

plt.legend(loc = 'upper right')



plt.subplot(1,2,2)

plt.ylabel('Accuracy', fontsize = 16)

plt.plot(hist.history['accuracy'], label = 'Training Accuracy')

plt.plot(hist.history['val_accuracy'], label = 'Validation Accuracy')

plt. legend(loc = 'lower right')

plt.show()
#training and fitting the model with validation

pred = model.predict(np.array(xtest))

pred
y_pred = [np.argmax(probas) for probas in pred]

print(y_pred)
# serialize model structure to JSON

from keras.models import model_from_json

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
with open('model.json', "r") as json_file:

    loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
