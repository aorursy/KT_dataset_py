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
import numpy as np
import pandas as pd
import random
from IPython.display import Image
from matplotlib import pyplot as plt

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D,Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
import keras.backend as K
from keras.models import Sequential
from keras import optimizers

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df_all = pd.read_csv("../input/whale-categorization-playground/train.csv", sep = ",")
input_path = "../input/whale-categorization-playground/train/train/"


size = 100 # image size
n_min = 20 # minimum number of images (5-60)


# filter out all whales with label "new_whale"
df = df_all[df_all.Id != 'new_whale']
# and filter out all whales with less than n_min images:
drop_list = []
for i in df['Id']:
    if df['Id'].value_counts()[i] < n_min:
        drop_list.append(i)
for i in drop_list:
    df = df[df.Id != i]

nw = len(df['Id'].value_counts()) # number of whales
mm = len(df) # number of images
print('\nWhale individuals: ', nw)
print('Number of images: ', mm)
print('\nNumber of images per whale individual: ')
print()
print(df['Id'].value_counts())
Image(filename=input_path+df['Image'][13]) 
X = df['Image']
y = df['Id']
def prepareImages(data):
    
    
    print("Preparing images")
    
    X = np.zeros((len(data), 100, 100, 3))
    
    count = 0
    
    for fig in data:
        #load images into images of size 100x100x3
        img = image.load_img(input_path+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X[count] = x
        if (count%10 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    count = 0
    
    print("Finished!")
            
    return X


def prepareY(Y):

    values = np.array(Y)
    print(values.shape)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

    y = onehot_encoded
    print(y.shape)
    return y, label_encoder
X = prepareImages(X)
X /= 255
y, label_encoder = prepareY(y)
plt.imshow(X[1]) 
mod = Sequential()

mod.add(Conv2D(32, (9, 9), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))
mod.add(Conv2D(64, (7, 7), strides = (1, 1), name = 'conv01'))

#mod.add(BatchNormalization(axis = 3, name = 'bn0'))
mod.add(Activation('relu'))

mod.add(MaxPooling2D((2, 2), name='max_pool'))
mod.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
mod.add(Activation('relu'))
mod.add(AveragePooling2D((3, 3), name='avg_pool'))

mod.add(Flatten())
mod.add(Dense(500, activation="relu", name='rl'))
mod.add(Dropout(0.8))
mod.add(Dense(y.shape[1], activation='softmax', name='sm'))

print(mod.output_shape)

#opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
mod.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
#mod.summary()
history = mod.fit(X, y, validation_split = 0.3, epochs=100, batch_size=100, verbose=2)
#plot how the accuracy changes as the model was trained
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
