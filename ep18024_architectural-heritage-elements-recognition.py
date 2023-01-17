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


import keras

from keras.preprocessing.image import array_to_img, img_to_array,  load_img



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPool2D

from tensorflow.keras.optimizers import Adam



from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten

from tensorflow.keras.utils import plot_model, to_categorical



import numpy as np

import pandas as pd

import re

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):

    return [os.path.join(root, f)

            for root, _, files in os.walk(directory) for f in files

            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]





X = []

Y = []





for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/0'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(0)





for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/1'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(1)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/2'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(2)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/3'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(3)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/4'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(4)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/5'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(5)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/6'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(6)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/7'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(7)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/8'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(8)



for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/train/9'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    X.append(img)

    Y.append(9)

# arrayに変換

X = np.asarray(X)

Y = np.asarray(Y)
test = []

for picture in list_pictures('../input/1056lab-archit-heritage-elem-recognit/test'):

    img = img_to_array(load_img(picture, target_size=(128,128)))

    test.append(img)



test = np.asarray(test)
X = X.astype('float32')

X = X / 255.0

test = test.astype('float32')

test= test / 255.0



Y= to_categorical(Y, 10)






# CNN

model = Sequential()



model.add(Conv2D(32,3,input_shape=(128,128,3)))

model.add(Activation('relu'))

model.add(Conv2D(32,3))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))





model.add(Conv2D(64,3))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(0.99))



model.add(Dense(10, activation='softmax'))



adam = Adam(lr=1e-4)



model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])



history = model.fit(X,Y, batch_size=512,epochs=2200,

                    verbose = 1)
# テストデータに適用

predict_classes = model.predict_classes(test)



submit_df = pd.read_csv('../input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv', index_col=0)

submit_df['class'] = predict_classes

submit_df.to_csv('Submission.csv')