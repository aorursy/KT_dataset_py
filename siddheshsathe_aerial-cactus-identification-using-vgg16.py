# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.

!unzip ../input/test.zip

!unzip ../input/train.zip
!mkdir has_cactus has_no_cactus 
df = pd.read_csv('../input/train.csv')
df.head()

import shutil

images_having_cactus = []

images_having_no_cactus = []



for i in df[df['has_cactus'] == 1]['id']:

    p = os.path.join('./train/', i)

    images_having_cactus.append(p)



for i in df[df['has_cactus'] == 0]['id']:

    p = os.path.join('./train/', i)

    images_having_no_cactus.append(p)



# Copying the images actually

for i in images_having_cactus:

    shutil.copy(i, './has_cactus/')

for i in images_having_no_cactus:

    shutil.copy(i, './has_no_cactus/')
print('Has Cactus: {}'.format(df[df['has_cactus'] == 1]['id'].count()))

print('Has No Cactus: {}'.format(df[df['has_cactus'] == 0]['id'].count()))
def augument_data(

    directory,              # Directory where augumentation is needed. Same dir will have sample images

    number_of_images_to_add # Image count to add 

):

    print('Images to add: {}'.format(number_of_images_to_add))

    import cv2

    from glob import glob

    l = glob(directory + '/*.jpg')

    for image in l:

        if number_of_images_to_add == 0:

            break

        img = cv2.imread(image)

        h_img = cv2.flip(img, 0)

        v_img = cv2.flip(img, 1)

        cv2.imwrite(directory + '/h_img_{}.jpg'.format(number_of_images_to_add), h_img)

        number_of_images_to_add -= 1

        cv2.imwrite(directory + '/v_img_{}.jpg'.format(number_of_images_to_add), v_img)

        number_of_images_to_add -= 1
augument_data('./has_no_cactus/', df[df['has_cactus'] == 1]['id'].count() - df[df['has_cactus'] == 0]['id'].count())
!mkdir -p curated_data/train_data curated_data/validation_data/has_cactus

!mkdir -p curated_data/validation_data/has_no_cactus

!mv has_cactus has_no_cactus curated_data/train_data
from glob import glob

import shutil

l = glob('curated_data/train_data/has_cactus/*.jpg')

for i in range(300):

    shutil.move(l[i], 'curated_data/validation_data/has_cactus')



l = glob('curated_data/train_data/has_no_cactus/*.jpg')

for i in range(300):

    shutil.move(l[i], 'curated_data/validation_data/has_no_cactus')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.callbacks import EarlyStopping
datagen = ImageDataGenerator(

    featurewise_std_normalization=True,

    samplewise_std_normalization=True,

    horizontal_flip=True,

    vertical_flip=True

)
train_data = datagen.flow_from_directory(

    'curated_data/train_data/',

    class_mode='categorical'

)



validation_data = datagen.flow_from_directory(

    'curated_data/validation_data/',

    class_mode='categorical'

)
vgg16_model = VGG16(

    include_top=False,

    weights='imagenet',

    input_shape=(256, 256, 3)

)
vgg16_model.summary()
for layer in vgg16_model.layers[:5]:

    layer.trainable = False
x = vgg16_model.output

x = Flatten()(x)

x = Dense(1024)(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

predictions = Dense(2, activation="softmax")(x)



# creating the final model 

model = Model(inputs= vgg16_model.input, outputs= predictions)



early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)



# compile the model 

model.compile(loss = "binary_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model.fit_generator(

    train_data,

    epochs=20,

    validation_data=validation_data,

    callbacks=[early_stop]

)
hist = pd.DataFrame(model.history.history)

hist.plot()
import cv2

from glob import glob

test_images = glob('../input/test/test/*.jpg')

df = pd.DataFrame(columns=['id', 'has_cactus'])

df.index.name = 'id'

for img in test_images:

    i = cv2.imread(img)

    i.resize(256, 256, 3)

    pred = model.predict(i.reshape(1, 256, 256, 3))

    tempDf = pd.DataFrame({

        'id': [img.split('/')[-1]],

        'has_cactus': [pred[0][0]]

    })

    df = df.append(tempDf)
# # Removing the new directories created locally

!rm -rf curated_data/
df = df.set_index('id')

df.to_csv('submission.csv')