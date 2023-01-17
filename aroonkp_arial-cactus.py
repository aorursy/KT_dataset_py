# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import cv2

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

import keras

from keras import layers

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
test_dir = '/kaggle/input/aerial-cactus-identification/test/test'

train_dir = '/kaggle/input/aerial-cactus-identification/train/train'

train_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
img1 = cv2.imread(train_dir+'/'+ train_df['id'][0])

plt.imshow(img1)
from tqdm import tqdm 
train_images = []



for img in train_df['id']:

    pather = train_dir + '/' +img

    train_images.append(cv2.resize(cv2.imread(pather),( 32, 32)))



X = np.asarray(train_images)

y = pd.DataFrame(train_df['has_cactus'])
X[:1]
y.head()
plt.imshow(X[0])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
input_sh = (32, 32, 3)
model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=input_sh))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



#model.add(Conv2D(32, (3, 3)))

#model.add(BatchNormalization())

#model.add(Activation('relu'))



model.add(Flatten())

#model.add(Dense(64))

#model.add(Activation('relu'))



#model.add(Dropout(0.6))



model.add(Dense(32))

model.add(Activation('relu'))

model.add(Dropout(0.6))



model.add(Dense(1))

model.add(Activation('sigmoid'))
opt = keras.optimizers.adam(lr=0.1)#, decay=1e-6)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator()
datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50),

                              steps_per_epoch=X_train.shape[0], 

                              epochs=1, validation_data=(X_test, y_test),

                              verbose=1)
test_images_names = []



for filename in tqdm(os.listdir(test_dir)):

    test_images_names.append(filename)
test_images_names[0]
test_dir
test_images_names[0]
#'/kaggle/input/aerial-cactus-identification/test/test/c662bde123f0f83b3caae0ffda237a93.jpg'
plt.imshow(cv2.imread(test_dir+'/'+test_images_names[0]))
test_images_names.sort()



images_test = []



for image_id in tqdm(test_images_names):

    pather = test_dir + '/' + image_id

    images_test.append((cv2.resize(cv2.imread(pather), (32, 32) )))

    

images_test = np.asarray(images_test)

images_test = images_test.astype('float32')

images_test /= 255
prediction = model.predict(images_test)
predict = []

for i in range(len(prediction)):

    if prediction[i][0]>0.5:

        answer = prediction[i][0]

    else:

        answer = prediction[i][0]

    predict.append(answer)



submission = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')



submission['has_cactus'] = predict
#sub = pd.DataFrame({'id' : np.array(test_images_names), 'has_cactus': np.array(predict)})
submission.to_csv('sample_submission.csv',index = False)
submission