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
import os

from tensorflow.python import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

import numpy as np

import pandas as pd
def convert_and_resize(img, to_array=True, color_mode='L', size=(512, 512)):

    img_ = img.resize(size)

    img_ = img_.convert(color_mode)

    if to_array:

        img_ = np.array(img_).astype(np.float32) / 255.

    return img_
img_size = (512, 512)
X_train = []

y_train = []
from tqdm import tqdm_notebook as tqdm 

from PIL import Image



for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class1')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class1', file_name)),

                                     size=img_size))

    y_train.append(0)

    

for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class1_def')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class1_def', file_name)),

                                      size=img_size))

    y_train.append(1)
for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class2')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class2', file_name)),

                                     size=img_size))

    y_train.append(0)

    

for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class2_def')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class2_def', file_name)),

                                      size=img_size))

    y_train.append(1)
for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class3')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class3', file_name)),

                                     size=img_size))

    y_train.append(0)

    

for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class3_def')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class3_def', file_name)),

                                      size=img_size))

    y_train.append(1)
for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class4')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class4', file_name)),

                                     size=img_size))

    y_train.append(0)

    

for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class4_def')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class4_def', file_name)),

                                      size=img_size))

    y_train.append(1)
for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class5')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class5', file_name)),

                                     size=img_size))

    y_train.append(0)

    

for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class5_def')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class5_def', file_name)),

                                      size=img_size))

    y_train.append(1)
for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class6')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class6', file_name)),

                                     size=img_size))

    y_train.append(0)

    

for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/train/Class6_def')):

    X_train.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/train/Class6_def', file_name)),

                                      size=img_size))

    y_train.append(1)
X_test = []

for file_name in tqdm(os.listdir('/kaggle/input/1056lab-defect-detection-extra/test')):

    X_test.append(convert_and_resize(Image.open(os.path.join('/kaggle/input/1056lab-defect-detection-extra/test', file_name)),

                                    size=img_size))
X_train = np.array(X_train).reshape(len(X_train), img_size[0], img_size[1], 1) 

X_test = np.array(X_test).reshape(len(X_test), img_size[0], img_size[1], 1)
from keras.utils.np_utils import to_categorical



y_train = np.array(y_train).reshape(-1, 1)

y_train = to_categorical(y_train, 2)
p = np.random.permutation(len(X_train))

X_train = X_train[p]

y_train = y_train[p]
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3) , activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten()) 

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(2, activation='sigmoid'))
from keras.callbacks import EarlyStopping, TensorBoard



es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

tb = TensorBoard(log_dir='./log', histogram_freq=1)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'] )
model.fit(X_train, y_train, validation_split=0.3, batch_size=32, epochs=5 )
predict = model.predict_proba(X_test, batch_size=32)[:, 1]
submit = pd.read_csv('/kaggle/input/1056lab-defect-detection-extra/sampleSubmission.csv')

submit['defect'] = predict

submit.to_csv('32-64-128.csv', index=False)