# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob

import os

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

figure(num=None, figsize=(16, 20), dpi=300, facecolor='w', edgecolor='k')

import tensorflow.keras as keras

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score

#import matplotlib.gridspec as gridspec

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#for dirname, _, filenames in os.walk('/kaggle/input/gtsrb-german-traffic-sign/Train/'):

#        print(dirname)

#train_images = glob.glob('/kaggle/input/landmark-recognition-2020/train/*/*/*/*')

train_csv = pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Train.csv')

test_csv = pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Test.csv')

train_csv.head(5)
training_images_path = np.array('/kaggle/input/gtsrb-german-traffic-sign/' + train_csv['Path'])

testing_images_path = np.array('/kaggle/input/gtsrb-german-traffic-sign/' + test_csv['Path'])
training_labels = np.array(train_csv['ClassId'])

testing_labels = np.array(test_csv['ClassId'])



training_labels = training_labels.reshape(-1)

testing_labels = testing_labels.reshape(-1)



print(training_labels.shape)

training_images_path,training_labels = shuffle(training_images_path,training_labels)



training_images_path = training_images_path

training_labels = training_labels



#unique_classes = pd.unique(training_labels)

#print('Total Classes = ',format(len(unique_classes)))

print('Total Training Examples = ',format(len(training_images_path)))
training_labels.shape
rand_num = np.random.randint(1 , 400,size = 25)

print(rand_num)

for num in range(len(rand_num)):

    plt.subplot(5,5,num+1)

    #print(training_images_path[rand_num[num]])

    img = cv2.imread(training_images_path[rand_num[num]])

    plt.imshow(img)

    plt.title(training_labels[rand_num[num]])

    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    plt.xticks([])

    plt.yticks([])
training_data = []

#print(training_images_path)

for num in range(len(training_images_path)):

    #print(num)

    img = cv2.imread(training_images_path[num],1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (30, 30),  

               interpolation = cv2.INTER_NEAREST)

    training_data.append(img)

    

training_data = np.array(training_data)
testing_data = []

for num in range(len(testing_images_path)):

    img = cv2.imread(testing_images_path[num],1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (30, 30),  

               interpolation = cv2.INTER_NEAREST)

    testing_data.append(img)

    

testing_data = np.array(testing_data)    
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=training_data.shape[1:]))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))



#Compilation of the model

model.compile(

    loss='sparse_categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy']

)
history = model.fit(training_data, training_labels, batch_size=32, epochs=30,

validation_split=0.4, verbose=2)
pred = model.predict_classes(testing_data)
accuracy_score(pred,testing_labels)
plt.figure(0)

plt.plot(history.history['accuracy'], label='training accuracy')

plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()



plt.figure(1)

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()