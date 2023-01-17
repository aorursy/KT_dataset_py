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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from cv2 import cv2
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout, Flatten, Input 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.utils import plot_model
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from numpy import array
df_train = pd.read_csv('/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv')
df_train.values

df_train.tail()
targets_series = pd.Series(df_train['level'])
one_hot = pd.get_dummies(targets_series, sparse = True)
targets_series[:10]

one_hot[:10]
one_hot_labels = np.asarray(one_hot)
one_hot_labelsY = np.asarray(targets_series)
one_hot_labelsY[:10]
im_size1 = 786
im_size2 = 786
x_train = []
y_train = []
i = 0 
for f, breed in tqdm(df_train.values):
    print(f)
df_test = df_train[:500]
"""
#this is a OpenCV implementation
i = 0 
for f, breed in tqdm(df_train.values):
    if type(cv2.imread('/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/{}.jpeg'.format(f)))==type(None):
        continue
    else:
        img = cv2.imread('/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/{}.jpeg'.format(f))
        label = one_hot_labels[i]
        x_train.append(cv2.resize(img, (im_size1, im_size2)))
        y_train.append(label)
        i += 1
np.save('x_train2',x_train)
np.save('y_train2',y_train)
print('Done')
"""
i=0
for f, breed in tqdm(df_test.values):
    try:
        img = image.load_img(('/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/{}.jpeg'.format(f)), target_size=(786, 786))
        arr = image.img_to_array(img)
        label = one_hot_labelsY[i]
        x_train.append(arr)
        y_train.append(label)
        i += 1 
    except:
        pass
len(x_train)
plt.imshow(x_train[400]/255) #681 > Try some other number too 
plt.show()
x_valid = []
y_valid = []
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
visible = Input(shape=(786,786,3))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
hidden1 = Dense(10, activation='relu')(flat)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
y_train_raw = np.array(Y_train)
x_train_raw = np.array(X_train)
model.summary()
model.fit(x_train_raw, y_train_raw, epochs=5)
x_valid_raw = np.array(X_valid)
y_valid_raw = np.array(Y_valid)
test_loss, test_acc = model.evaluate(x_valid_raw, y_valid_raw)
test_loss
test_acc
