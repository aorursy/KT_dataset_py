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
import pandas as pd

import os,cv2

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import DenseNet121, DenseNet169, DenseNet201

from keras.models import Sequential, Model

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split



import numpy as np
train_dir = '../input/train'

test_dir = '../input/test'

submission = pd.read_csv('../input/sample_submission.csv')
categories = os.listdir(train_dir)

categories
i=0

images = []

for i in range(0,12):

    image = os.listdir(os.path.join(train_dir,categories[i]))

    img = cv2.imread(train_dir + '/' + categories[i] + '/' + image[i])

    images.append(img)

    i = i+1

    

plt.figure(figsize=[10,10])

for x in range(0,12):

    plt.subplot(3, 4,x+1)

    plt.imshow(images[x])

    plt.title(categories[x])

    x += 1

    

plt.show()
len(images)
train = []

for species_id, sp in enumerate(categories):

    for file in os.listdir(os.path.join(train_dir, sp)):

        train.append(['train/{}/{}'.format(sp, file), file, species_id, sp])

train_df = pd.DataFrame(train, columns=['filepath', 'file', 'species_id', 'species'])



print('train_df.shape = ', train_df.shape)



# read all test data

test = []

for file in os.listdir(test_dir):

    test.append(['../input/test/{}'.format(file), file])

test_df = pd.DataFrame(test, columns=['filepath', 'file'])

print('test_df.shape = ', test_df.shape)
train_df.head()

train_df = train_df.sample(frac=1).reset_index(drop=True)



train_df.head()
from keras.preprocessing import image
data_dir = '../input'

def read_img(filepath, size):

    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)

    img = image.img_to_array(img)

    return img
test_df.head()

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,)





test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    directory = '../input/train',

    target_size = (256,256),

    color_mode="rgb",

    batch_size=32,

    class_mode="categorical",

    shuffle=True,

    seed=1978

    )



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(256, 256, 3), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(12, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
history = model.fit_generator(train_generator,

steps_per_epoch = 1000,

epochs = 25,

)
model.save('ResNet50Keras.h5')
test_generator = test_datagen.flow_from_dataframe(

test_df,

directory = None,

x_col = 'filepath',

y_col = None,

class_mode = None,

target_size = (256,256),

batch_size = 64

)

predict = model.predict_generator(test_generator,steps=np.ceil(test_df.shape[0]/64))

len(predict)
submission['file'].shape

test_df['file'].shape
predict[0]
pred = np.argmax(predict,axis=1)



pred_class = []

for i in pred:

    pred_class.append(categories[pred[i]])

    
len(pred_class)

pred_class
result = {'file':test_df['file'],'species':pred_class}

result = pd.DataFrame(result)

result.to_csv("submission.csv",index=False)
result.head()