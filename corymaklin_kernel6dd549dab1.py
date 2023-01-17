# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2

from keras.models import Sequential

from keras.layers import Flatten, Conv2D, MaxPool2D, Activation, Dense, Dropout

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
df = pd.read_csv('../input/train.csv')

df.head()
train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.15,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)
df['has_cactus'] = df['has_cactus'].astype(str)



train_generator = train_datagen.flow_from_dataframe(

    df,

    directory = '../input/train/train',

    subset = 'training',

    x_col = 'id',

    y_col = 'has_cactus',

    target_size = (32,32),

    class_mode = 'binary'

)

test_generator = train_datagen.flow_from_dataframe(

    df,

    directory = '../input/train/train',

    subset = 'validation',

    x_col = 'id',

    y_col = 'has_cactus',

    target_size = (32,32),

    class_mode = 'binary'

)
model = Sequential()



model.add(Conv2D(32, (3,3) ,activation = 'relu', input_shape = (32,32,3)))

model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPool2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPool2D(2,2))



model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))
model.compile(

    loss = 'binary_crossentropy',

    optimizer = Adam(), 

    metrics = ['accuracy']

)
history = model.fit_generator(

    train_generator,

    steps_per_epoch = 2000,

    epochs = 10,

    validation_data = test_generator,

    validation_steps = 64

)
ids = []

X_test = []



for image in os.listdir(test_directory):

    ids.append(image.split('.')[0])

    path = os.path.join(test_directory, image)

    X_test.append(cv2.imread(path))

    

X_test = np.array(X_test)

X_test = X_test.astype('float32') / 255
predictions = model.predict(X_test)
submission = pd.read_csv('sample_submission.csv')

submission['has_cactus'] = predictions

submission['id'] = ids

submission.head(10)
submission.to_csv('submission.csv', index = False)