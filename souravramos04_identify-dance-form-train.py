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
from keras.layers import Input, Lambda, Dense, Flatten, Dropout

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

from glob import glob

import matplotlib.pyplot as plt

import os

import cv2
df = pd.read_csv('../input/identifythedanceform/train.csv')
df.head()
# resize all the images to this

IMAGE_SIZE = [224, 224]
train_path = '../input/identifythedanceform/train'

test_path = '../input/identifythedanceform/test'

df['target'].value_counts()
images = []

labels = list(df['target'])

for filename in list(df['Image']):

    image = cv2.imread(os.path.join(train_path, filename))

    image = cv2.resize(image, (224,224))

    image = preprocess_input(image)

    images.append(image)

    
images = np.array(images)

labels = np.array(labels)
labels
lb = LabelEncoder()

labels = lb.fit_transform(labels)

print(labels[:10])

labels = to_categorical(labels)

print(labels[:10])
(trainX, testX, trainY, testY) = train_test_split(images, labels,

                                test_size=0.20, stratify=labels, random_state=42)
print(len(trainX))

print(len(trainY))

print(len(testX))

print(len(testY))

testX.shape
data_gen = ImageDataGenerator(

    rescale = 1/255.0,

    rotation_range = 20,

    zoom_range = 0.2,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.2,

    fill_mode = 'nearest',

    horizontal_flip = True

)
# add preprocessing layer to the fromt of VGG

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# don't train existing weights

for layer in vgg.layers:

    layer.trainable = False
# our layers - you can add more if you want

x = Flatten()(vgg.output)

x = Dense(1024, activation = 'relu')(x)

x = Dropout(0.20)(x)

prediction = Dense(8, activation='softmax')(x)
# create a model object

model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model

model.summary()
model.compile(

    loss='categorical_crossentropy',

    optimizer = 'adam',

    metrics = ['accuracy']

)
train_generator = data_gen.flow(trainX, trainY, batch_size=32)

test_generator = data_gen.flow(testX, testY, batch_size = 32)
# Set a Learning Rate Annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',

                                           patience=3,

                                           verbose=1,

                                           factor=0.5,

                                           min_lr=0.00001)
history = model.fit_generator(train_generator,

        validation_data = test_generator,

        epochs = 100,

        steps_per_epoch = len(train_generator),

        validation_steps = len(test_generator),

        callbacks = [learning_rate_reduction]

)
# loss

plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.legend()

plt.show()

plt.savefig('LossVal_loss1')



# accuracies

plt.plot(history.history['accuracy'], label='train acc')

plt.plot(history.history['val_accuracy'], label='val acc')

plt.legend()

plt.show()

plt.savefig('AvvVal_acc1')
test_df = pd.read_csv('../input/identifythedanceform/test.csv')
test_df.head()
test_images = []

for filename in list(test_df['Image']):

    image = cv2.imread(os.path.join(test_path, filename))

    image = cv2.resize(image, (224,224))

    image = preprocess_input(image)

    test_images.append(image)
test_images = np.array(test_images)



test_images.shape
test_labels = []
# create test labels 

def create_test_labels():

    for image in test_images:

        image = cv2.resize(image, (224, 224))

        image = image.reshape(-1,224,224,3)

        image = preprocess_input(image)

        predict = model.predict(image)

        test_labels.append([image, predict])



create_test_labels()

Image = []

target = []

for i, j in test_labels:

    target.append(np.argmax(j))

    Image.append(i)

df = pd.DataFrame(columns=['target'])

df['target'] = target

df['target'] = lb.inverse_transform(df['target'])

datasets = pd.concat([test_df['Image'], df['target']], axis=1)

datasets.to_csv('sample_submission.csv', index=False)
datasets