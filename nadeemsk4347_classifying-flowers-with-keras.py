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
import os
import numpy as np
import scipy.io
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
img_labels = scipy.io.loadmat("/kaggle/input/flower-dataset-102/imagelabels.mat")
img_labels = img_labels["labels"]
img_labels = img_labels[0]
for i in range(len(img_labels)):
  img_labels[i] = img_labels[i] - 1
print(img_labels)
import tarfile
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_SIZE = 50

train_x = []
train_y = []
tar = tarfile.open('/kaggle/input/flower-dataset-102/102flowers.tgz', "r:gz")
i = 0
for tarinfo in tqdm(tar):
    i+=1
    tar.extract(tarinfo.name)
    
    if(tarinfo.name[-4:] == '.jpg'):
        var = tarinfo.name[11:15]
        img_num = int(var)-1
        train_y.append(img_labels[img_num])
        
        image = cv2.imread(tarinfo.name)
        resized = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
        normalized_img = cv2.normalize(resized, None, alpha=0, beta=1, 
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        train_x.append(normalized_img)

#         label_list.append(tarinfo.name.split('_')[0])
    if(tarinfo.isdir()):
        os.rmdir(tarinfo.name)
    else:
        os.remove(tarinfo.name) 

tar.close()
train_x = np.array(train_x)
plt.imshow(train_x[1002])
print(train_y[1002])
print(train_x.shape)
trainx, testx, trainy, testy = train_test_split(train_x, train_y, test_size=0.10, random_state=10)

trainx, valx, trainy, valy = train_test_split(trainx, trainy, test_size=0.15, random_state=10)

trainy = to_categorical(trainy)
testy = to_categorical(testy)
valy = to_categorical(valy)
np.save('testx.npy', testx)
np.save('testy.npy', testy)

print("Training data number:",len(trainx))
print("Testing data number:",len(testx))
print("Validation data number:",len(valx))

print("Training labels number:",len(trainy))
print("Testing labels number:",len(testy))
print("Validation labels number:",len(valy))
print("Training data shape:",trainx.shape)
print("Testing data shape:",testx.shape)
print("Validation data shape:",valx.shape)

print("Training labels shape:",trainy.shape)
print("Testing labels shape:",testy.shape)
print("Validation labels shape:",valy.shape)
"""## Creating Model"""

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(102, activation='softmax'))


# model = Sequential()
# #Layer 1
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))
# #Layer 2
# model.add(Conv2D(128, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))
# #Layer 3
# model.add(Conv2D(128, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))
# #Layer 4
# model.add(Conv2D(256, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D((2,2)))
# #Input to Neural Network is flattened
# model.add(Flatten())
# #1st hidden layer with 512 neurons/nodes
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# #Output layer with 102 nodes for classifying 102 flowers
# model.add(Dense(102, activation='softmax'))

"""## Compiling and Training the Neural Network"""

#Compile the neural network
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Train the network
model.fit(trainx, trainy, validation_data = (valx, valy), epochs=60, batch_size=20)
model.save('model.h5')
from keras.models import load_model
model = load_model("model.h5")

score = model.evaluate(testx, testy)

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

#Predict output on sample input data
pred = model.predict(testx) 
pred = np.argmax(pred, axis = 1)[:10] 
label = np.argmax(testy,axis = 1)[:10] 

print("Predicted labels:",pred) 
print("Actual Labels:   ",label)