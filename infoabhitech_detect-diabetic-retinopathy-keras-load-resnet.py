# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/aptos2019-blindness-detection"))

print(os.listdir("../input/resnet-model"))



import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
import keras

from keras.models import load_model

from keras import optimizers

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical



import matplotlib.pyplot as plt

%matplotlib inline



from tqdm import tqdm
# Loading resnet50 model

res_model = load_model('../input/resnet-model/resnet_model.h5')
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train['diagnosis'].head()
# Function to generate image file

def train_func_image_file(x):

    folder = '../input/aptos2019-blindness-detection/train_images/'

    path = folder + x + '.png'

    return path
train['path'] = train['id_code'].apply(train_func_image_file)
train.head()
input_shape = (128, 128, 3)
# Transforming high resolution images to low resolution to meet up memory constraints

train_image = []

for i in tqdm(range(train.shape[0])):

    img = image.load_img(train['path'][i],target_size=input_shape,interpolation='nearest')

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

x_train = np.array(train_image)
print("Image diagnosed with class 4 retinopathy",plt.imshow(x_train[1]))
print("Image for normal eye",plt.imshow(x_train[3]))
x_train.shape
y_train = train['diagnosis']

y_train = keras.utils.to_categorical(y_train, 5)
y_train.shape
del train, train_image

import gc; 

gc.collect()
batch_size = 32

epochs = 3
model = Sequential()

model.add(res_model)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()
model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs, verbose=1)

# validation_split=0.2
del x_train, y_train

import gc; 

gc.collect()
# Test data preparation for competition submission

# Create list of test image files

image_file = []

for file in os.listdir("../input/aptos2019-blindness-detection/test_images/"):

    image_file.append(file)
# Create test data frame

test = pd.DataFrame(image_file,columns=['file'])
# Function to generate image test file

def test_func_image_file(x):

    folder = '../input/aptos2019-blindness-detection/test_images/'

    path = folder + x

    return path
test['path'] = test['file'].apply(test_func_image_file)
test_image = []

for i in tqdm(range(test.shape[0])):

    img = image.load_img(test['path'][i],target_size=input_shape,interpolation='nearest')

    img = image.img_to_array(img)

    img = img/255

    test_image.append(img)

x_test = np.array(test_image)
predictions = model.predict_classes(x_test)
del x_test,test_image

import gc; 

gc.collect()
test['id_code'] = test['file'].apply(lambda x: os.path.splitext(x)[0])
test['diagnosis'] = pd.Series(predictions)
test = test.drop(['file','path'], axis=1)
test.head()
test['diagnosis'].unique()
test.to_csv("submission.csv", columns = test.columns, index=False)