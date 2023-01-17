# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.datasets import mnist

from sklearn.utils import class_weight, shuffle
os.listdir('/kaggle/input/animals10/animals/raw-img')

foldernames = os.listdir('/kaggle/input/animals10/animals/raw-img')

categories = []

files = []

i = 0

labels = []

for folder in foldernames:

#     j = 0

    filenames = os.listdir("../input/animals10/animals/raw-img/" + folder);

    for file in filenames:

#         if (j == 200): break

        files.append("../input/animals10/animals/raw-img/" + folder + "/" + file)

        categories.append(i)

#         j+=1

    i = i + 1

    labels.append(folder)

        

        

df = pd.DataFrame({

    'filename': files,

    'category': categories

})
x = df['filename']

y = df['category']



x, y = shuffle(x, y, random_state=8)

y.hist()
import cv2

sift = cv2.ORB_create()

def fd_sift(image) :

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    kps, des = sift.detectAndCompute(image, None)

    return des if des is not None else np.array([]).reshape(0, 128)
global_features = []

labels          = y

fixed_size = (500,500)

for file in x[:2000]:

    image = cv2.imread(file)

    image.resize((500,500,3))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     fv_sift = fd_sift(image)

#     global_feature = np.hstack([fv_sift])

    global_feature = np.hstack([image])

    global_feature.resize(fixed_size)

    global_features.append(global_feature)
for i in global_features[:10]:

    print(i.shape)
X_train = np.array(global_features)

X_train = X_train.reshape(len(X_train),500*500)

Y_train = np.array(y[0:2000])

len(X_train),X_train.shape
from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, 

                                                      test_size=0.3, 

                                                      stratify=Y_train, 

                                                      random_state=8)

print(train_x.shape)

print(train_y.shape)

print(valid_x.shape)

print(valid_y.shape)
(X_train, y_train), (X_test, y_test) = (train_x, train_y), (valid_x, valid_y)  # mnist.load_data()

test_x,X_val, test_y, y_val = train_test_split(X_test, y_test, 

                                                      test_size=0.5, 

                                                      stratify=y_test, 

                                                      random_state=8)

# X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]

# X_train, y_train = X_train[:50000,:], y_train[:50000]

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
X_train = X_train.reshape(X_train.shape[0], 500, 500,1)

X_val = X_val.reshape(X_val.shape[0], 500, 500,1)

X_test = X_test.reshape(X_test.shape[0], 500, 500,1)

print(X_train.shape,X_val.shape,X_test.shape)
Y_train = np_utils.to_categorical(y_train,10)

Y_val = np_utils.to_categorical(y_val,10)

Y_test = np_utils.to_categorical(y_test,10)

print(Y_train.shape,Y_val.shape,Y_test.shape)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(500,500,1)))

model.add(Conv2D(32, (3, 3), activation='sigmoid'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='sigmoid'))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=10, verbose=1)
fig = plt.figure()

numOfEpoch = 10

plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')

plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')

plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')

plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')

plt.title('Accuracy and Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss|Accuracy')

plt.legend()
score = model.evaluate(X_test, Y_test, verbose=0)

print(score)