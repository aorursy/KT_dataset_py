# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print('----')

    i = 0

    for filename in filenames:

        print(os.path.join(dirname, filename))

        i += 1

        if i > 5:

            break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"

TEST_PATH = "../input/plant-pathology-2020-fgvc7/test.csv"

TRAIN_PATH = "../input/plant-pathology-2020-fgvc7/train.csv"

SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"



sub = pd.read_csv(SUB_PATH)

test_data = pd.read_csv(TEST_PATH)

train_data = pd.read_csv(TRAIN_PATH)
train_data.head()
train_paths = IMAGE_PATH + train_data['image_id'].values + '.jpg'

train_paths
def load_image(image_file,size=(256,256)):

    image = cv2.imread(image_file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return cv2.resize(image, dsize=size)
X = np.array([*map(lambda x: load_image(x), train_paths)], dtype=np.float32) / 255
y = train_data.iloc[:,1:5].values

y
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=2020)
datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

datagen.fit(X_train)
from tensorflow.keras import layers, models
model = models.Sequential()

model.add(layers.Conv2D(64,(3,3),strides=2,activation='relu',input_shape=(256,256,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128,(3,3),strides=2,activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dropout(0.25))

model.add(layers.BatchNormalization())

model.add(layers.Dense(4,activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train) / 32, epochs=5, validation_data=(X_valid,y_valid))
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid,y_valid))
test_paths = IMAGE_PATH + test_data['image_id'].values + '.jpg'

test = np.array([*map(lambda x: load_image(x), test_paths)], dtype=np.float32) / 255
predict = model.predict(test)
submission = pd.DataFrame(predict)

submission.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

submission['image_id'] = sub['image_id']

submission = submission[['image_id','healthy', 'multiple_diseases', 'rust', 'scab']]

submission.to_csv('submission.csv',index=False)

submission