# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from shutil import copyfile

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib

import os

from shutil import copyfile

import matplotlib.pyplot as plt
labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels.head()
labels_dict = {i:j for i,j in zip(labels['id'],labels['breed'])}

classes = set(labels_dict.values())

classes
images = [f for f in os.listdir('/kaggle/input/dog-breed-identification/train/')]
split = int(len(images) * 0.85)
training_images = images[:split]

validation_images  = images[split:]
if  not os.path.exists('training_images'):

        os.makedirs('training_images')



if  not os.path.exists('validation_images'):

    os.makedirs('validation_images')
for curClass in classes:    

    if  not os.path.exists(os.path.join('training_images', curClass)):

        os.makedirs(os.path.join('training_images', curClass))
for curClass in classes:    

    if  not os.path.exists(os.path.join('validation_images', curClass)):

        os.makedirs(os.path.join('validation_images', curClass))
count = 0 

destination_directory = 'training_images/'

for item in images:

    if count >7999:

        destination_directory = 'validation_images/'

    filekey = os.path.splitext(item)[0]

    des = destination_directory + labels_dict[filekey]+'/'+item

    print(des)

    if  not os.path.exists(des):

        src = '/kaggle/input/dog-breed-identification/train/' + item

        copyfile(src, des)

    print(labels_dict[filekey])

    count +=1
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img = load_img('training_images/cocker_spaniel/dd126e42b474c3831f8fda33052428c1.jpg') 

datagen = ImageDataGenerator(

        rotation_range=50,

        width_shift_range=0.3,

        height_shift_range=0.2,

        shear_range=0.3,

        zoom_range=0.3,

        horizontal_flip=True,

        fill_mode='nearest')
img = load_img('/kaggle/input/dog-breed-identification/train/09839ef1c5a5a5b3acb61c4093cab07f.jpg') 

x = img_to_array(img)

x = x.reshape((1,) + x.shape)
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Conv2D,Dropout

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(

        'training_images',

        target_size=(128, 128),

        batch_size=20,

        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(

        'training_images',

        target_size=(128, 128),

        batch_size=20,

        class_mode='categorical')
test_set = test_datagen.flow_from_directory(

        'validation_images',

        target_size=(128, 128),

        batch_size=20,

        class_mode='categorical')
from keras.layers import Dropout

clf = Sequential()

#Convolution

#32 is number of kernals of 3x3, we can use 64 128 256 etc in next layers

#input shape can be 128, 256 later

clf.add(Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu'))

#Max Pooling size reduces divided by 2

clf.add(MaxPooling2D(pool_size=(2,2)))      





#clf.add(Dropout(0.5))



clf.add(Conv2D(32,(3,3), activation='relu'))

clf.add(MaxPooling2D(pool_size=(2,2)))

#clf.add(Dropout(0.25))



clf.add(Conv2D(64, (3, 3), activation='relu'))

clf.add(MaxPooling2D(pool_size=(2, 2)))

#clf.add(Dropout(0.10))

#Flattening

clf.add(Flatten())

        

#Adding An ANN

#lets take 128 hidden nodes in hidden layer

#clf.add(Dense(units=128,activation='relu'))

clf.add(Dense(units=64, activation='relu'))

clf.add(Dropout(0.5))

clf.add(Dense(units=120,activation='softmax'))

#stochastic gradient descent -Adam -optimizer

#loss func categorical cross entropy

#metrics = accuracy

clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.callbacks import EarlyStopping

early_stopping_monitor=EarlyStopping(patience=6)
hist=clf.fit_generator(

        training_set,

        steps_per_epoch=400,

        epochs=50,

        validation_data=test_set,

        validation_steps=2222,

callbacks=[early_stopping_monitor])
import os

import cv2

import pandas as pd

test_set = []

test_set_ids = []



for curImage in os.listdir('/kaggle/input/dog-breed-identification/test'):

    test_set_ids.append(os.path.splitext(curImage)[0])

    #print(os.path.splitext(curImage)[0])

    curImage = cv2.imread('/kaggle/input/dog-breed-identification/test/'+curImage)

    test_set.append(cv2.resize(curImage,(128, 128)))
test_set = np.array(test_set, np.float32)/255.0
predictions= clf.predict(test_set)
predictions[0].shape
training_set.class_indices
classes= {index:breed for breed,index in training_set.class_indices.items()}

column_names = [classes[i] for i in range(120)]

column_names
predictions_df = pd.DataFrame(predictions)

predictions_df.columns = column_names

predictions_df.insert(0,'id', test_set_ids)



predictions_df
predictions_df.to_csv('submission.csv')
plt.plot(hist.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('validation loss')

plt.show()
plt.plot(hist.history['loss'],label="traing loss")

plt.plot(hist.history['val_loss'], label="Validation loss")

plt.legend()

plt.xlabel('epochs')

plt.show()