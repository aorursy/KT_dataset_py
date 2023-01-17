# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train['Id'].describe()
y_train = train['Id']
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



def prepareImages(train, shape, path):

    

    x_train = np.zeros((shape, 100, 100, 3))

    count = 0

    

    for fig in train['Image']:

        

        #load images into images of size 100x100x3

        img = image.load_img("../input/"+path+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        x_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return x_train
X_train = prepareImages(train, train.shape[0], 'train')

X_train/=255
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)

y_train = to_categorical(y_train, num_classes = 5005)
y_train.shape
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dropout, Flatten, MaxPooling2D, Conv2D, Dense

from keras.layers.normalization import BatchNormalization
model = Sequential()



model.add(Conv2D(32, (5,5), strides = (1,1), padding='same', activation = 'relu', input_shape = (100, 100, 3)))

model.add(Conv2D(32, (5,5), strides = (1,1), padding = 'same', activation='relu'))

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(32, (3,3), strides = (2,2), padding='same', activation='relu'))

model.add(Conv2D(32, (3,3), strides = (2,2), padding='same', activation='relu'))

model.add(MaxPooling2D((2,2), strides = (2,2)))



model.add(Conv2D(64, (3,3), strides = (1,1), padding='same', activation='relu'))

model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))

model.add(MaxPooling2D((2,2), strides = (2,2)))



model.add(Dropout(0.2))

model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))

model.add(Dense(y_train.shape[1], activation = 'softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 100

batchsize = 1024
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batchsize, verbose=2)
plt.plot(history.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
plt.plot(history.history['acc'], color='g', label="Train Accuracy")

plt.title("Train Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print('Train accuracy of the model: ',history.history['acc'][-1])
test = os.listdir("../input/test/")

print(len(test))
test_data = pd.DataFrame(test, columns=['Image'])

test_data['Id'] = ''
X_test = prepareImages(test_data, test_data.shape[0], "test")

X_test /= 255
predictions = model.predict(np.array(X_test), verbose=1)
for i, pred in enumerate(predictions):

    test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_data.to_csv('submission_1.csv', index=False)