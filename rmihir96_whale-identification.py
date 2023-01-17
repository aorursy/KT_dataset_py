# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow

from sklearn.model_selection import train_test_split



from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D



from keras.utils import np_utils

from keras import backend as K

#K.set_image_dim_ordering('th')

# Any results you write to the current directory are saved as output.
#fixing a random seed

seed = 7

np.random.seed(seed)
training_data = pd.read_csv("../input/train.csv")

training_data
y = training_data['Id']

X = training_data['Image']
print(X.shape)

print(y.shape)
from matplotlib.image import imread

count = 0

train_X = np.zeros((X.shape[0], 100, 100, 3))

print(train_X.shape)

for i in X:

    

    #load images into images of size 100x100x3

    img = image.load_img("../input/train/" + i , target_size=(100, 100, 3))

    x = image.img_to_array(img)

    x = preprocess_input(x)

   # print(x.shape)

    train_X[count] = x

    if (count%500 == 0):

        print("Processing image: ", count+1, ", ", i)

    count += 1



print(train_X.shape)

train_X/=255
testing_data = os.listdir("../input/test/")

print(len(testing_data))



n_test = len(testing_data)


count = 0

test_imgs = np.zeros((n_test, 100, 100, 3))

print(test_imgs.shape)

for i in testing_data:

    

    #load images into images of size 100x100x3

    img = image.load_img("../input/test/" + i , target_size=(100, 100, 3))

    x = image.img_to_array(img)

    x = preprocess_input(x)

   # print(x.shape)

    test_imgs[count] = x

    if (count%500 == 0):

        print("Processing image: ", count+1, ", ", i)

    count += 1



print(test_imgs.shape)



test_imgs /= 255

onehot = pd.get_dummies(y)

#print(onehot)

target_labels = onehot.columns

print(target_labels)

y = onehot.as_matrix()

print(y.shape)

n = y.shape[1]
target_labels[2]
from PIL import Image

img = imread("../input/train/659583f73.jpg" )

print(img.shape)



image = Image.open("../input/train/659583f73.jpg")

plt.imshow(image)
#Create model



model = Sequential()

model.add(Convolution2D(32, (5,5),  input_shape=(100,100,3), activation='relu'))

model.add(Dropout(0.2))

model.add(Convolution2D(32, (5,5), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (5,5), activation = 'relu'))

model.add(Dropout(0.2))

model.add(Convolution2D(64, (5,5), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, (5,5), activation = 'relu'))

model.add(Dropout(0.2))

model.add(Convolution2D(128, (5,5), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(1024, activation= 'relu' ))

model.add(Dropout(0.2))

model.add(Dense(512, activation= 'relu' ))

model.add(Dropout(0.2))

model.add(Dense(n, activation= 'softmax' ))



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()




history = model.fit(train_X, y, epochs=25, batch_size=64, verbose=1)

# Final evaluation of the model

#scores = model.evaluate(X_test, y_test, verbose=0)

#print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
col = ['Image']

test_df = pd.DataFrame(testing_data, columns=col)

test_df
predictions = model.predict(np.array(test_imgs), verbose=1)
results = []

for i in predictions:

    results.append(np.argmax(i))



print(len(results))

test_df['Id'] = results

test_df.head()
final_pred = []

for x in test_df.Id:    

    #print(x)

    #print(target_labels[x])

    final_pred.append(target_labels[x])

print(len(final_pred))



test_df['Id'] = final_pred

test_df.head()
test_df.to_csv('submission.csv', index=False)
test_df.Id.value_counts()

