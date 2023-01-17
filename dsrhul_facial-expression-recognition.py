# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from numpy import asarray



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/facial-expression/fer2013.csv")

output_label = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("Unique labels:",len(np.unique(np.array(df['emotion']))))

print("Pixel type:",type(df['pixels'][2]))

df.tail(5)
#number of samples per label

sns.countplot(x=df['emotion'], data=df)
#Convert the df['pixels'] str type to array and reshape

def string2array(x):

  return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')



X= df['pixels'].apply(lambda x: string2array(x))

X = np.array(X)

X = np.stack(X, axis = 0)

X = X/255.0

y = np.array(df['emotion'])
#sample image

img_no = 1246 #image number range(0,35k)

img = X[img_no].reshape(48,48)

plt.figure()

plt.title(output_label[y[img_no]])

plt.imshow(img, cmap = 'gray')
#convert numpy array to categorical 

from keras.utils import to_categorical

y = to_categorical(y)

print(X.shape,y.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, shuffle = True)
print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
from keras.layers import Dense

from keras.layers import Conv2D, Activation

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization



#defining layers

num_classes = 7

model = Sequential()

input_shape = (48,48,1)

model.add(Conv2D(64, (4, 4), activation='relu', padding='same'))

model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (4, 4),activation='relu',padding='same'))

model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(7))

model.add(Activation('softmax'))

    

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    validation_split = 0.25,

    fill_mode="nearest")
model.fit(aug.flow(x_train, y_train, batch_size=64),

            epochs=20, 

            verbose=1, 

            shuffle=True,

            )
preds = model.predict(x_test, verbose=1)
#model performance evaluation

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy score: ",accuracy_score(y_test, np.round_(preds)))

print("Classification report:")

print(classification_report(y_test, np.round_(preds)))
#sample image

img_no = 6 #image number range(0,8972)

img = x_test[img_no].reshape(48,48)

plt.figure()

plt.title(output_label[np.argmax(preds[img_no])])

plt.imshow(img, cmap='gray')