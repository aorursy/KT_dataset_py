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
os.listdir()
os.listdir('../')
os.listdir('../input/')
os.listdir('../input/aerial-cactus-identification')
os.listdir('../input/aerial-cactus-identification/train/train')
train_dir = '../input/aerial-cactus-identification/train/train'
csv_path = '../input/aerial-cactus-identification/train.csv'

df = pd.read_csv(csv_path)
df.head()
filenames = df['id']

filenames.head()
file_paths =[os.path.join(train_dir, fname) for fname in filenames]

file_paths[:5]
train_df = pd.DataFrame(data ={'id':file_paths, 'has_cactus': df['has_cactus']})

train_df.head()
train_df = train_df.astype(np.str)
train_df.head()
sample_csv_path = '../input/aerial-cactus-identification/sample_submission.csv'

sample_df = pd.read_csv(sample_csv_path)

sample_df.head()
len(train_df)
train_df = train_df[:-500]

test_df = train_df[-500:]

len(train_df), len(test_df)
path = train_df['id'][0]
from tqdm import tqdm_notebook



import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir
test_dir = '../input/aerial-cactus-identification/test/test'
os.listdir(test_dir)
len(test_dir)
path
img_pil=Image.open(path)

image = np.array(img_pil)

image.shape
plt.imshow(image)

plt.show()
input_shape = (32,32,3)

batch_size = 32

num_classes = 2

num_epochs =1

learning_rate = 0.01
inputs = layers.Input(input_shape)

net = layers.Conv2D(64, (3, 3), padding='same')(inputs)

net = layers.Conv2D(64, (3, 3), padding='same')(net)

net = layers.Conv2D(64, (3, 3), padding='same')(net)

net = layers.BatchNormalization()(net)

net = layers.Activation('relu')(net)

net = layers.MaxPooling2D(pool_size=(2, 2))(net)



net = layers.Conv2D(128, (3, 3), padding='same')(net)

net = layers.Conv2D(128, (3, 3), padding='same')(net)

net = layers.Conv2D(128, (3, 3), padding='same')(net)

net = layers.BatchNormalization()(net)

net = layers.Activation('relu')(net)

net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Dropout(0.25)(net)



net = layers.Conv2D(256, (3, 3), padding='same')(net)

net = layers.Conv2D(256, (3, 3), padding='same')(net)

net = layers.Conv2D(256, (3, 3), padding='same')(net)

net = layers.BatchNormalization()(net)

net = layers.Activation('relu')(net)

net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Dropout(0.25)(net)



net = layers.Conv2D(512, (3, 3), padding='same')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)

net = layers.BatchNormalization()(net)

net = layers.Activation('relu')(net)

net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Dropout(0.25)(net)



net = layers.Conv2D(512, (3, 3), padding='same')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)

net = layers.BatchNormalization()(net)

net = layers.Activation('relu')(net)

net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Dropout(0.25)(net)



net = layers.Flatten()(net)

net = layers.Dense(512)(net)

net = layers.Activation('relu')(net)

net = layers.Dropout(0.5)(net)

net = layers.Dense(num_classes)(net)

net = layers.Activation('softmax')(net)



model = tf.keras.Model(inputs=inputs, outputs=net)
model.compile(loss='sparse_categorical_crossentropy',

              optimizer=tf.keras.optimizers.Adam(learning_rate),

              metrics=['accuracy'])
train_datagen = ImageDataGenerator(

    rescale=1./255.,

    width_shift_range=0.3,

    zoom_range=0.2,

    horizontal_flip=True

)



test_datagen = ImageDataGenerator(

    rescale=1./255.

)
train_generator = train_datagen.flow_from_dataframe(

    train_df,

    x_col='id',

    y_col='has_cactus',

    target_size=input_shape[:2],

    batch_size=batch_size,

    class_mode='sparse'

)



test_generator = test_datagen.flow_from_dataframe(

    test_df,

    x_col='id',

    y_col='has_cactus',

    target_size=input_shape[:2],

    batch_size=batch_size,

    class_mode='sparse'

)
model.fit_generator(

    train_generator,

    steps_per_epoch=len(train_generator),

    epochs=num_epochs,

    validation_data=test_generator,

    validation_steps=len(test_generator)

)
test_dir
sample_df.head()
path = os.path.join(test_dir, sample_df['id'][0])
img_pil = Image.open(path)

image = np.array(img_pil) # np.array형으로 이미지를 받아옴.

image.shape
plt.imshow(image)

plt.show()
print(image.dtype)

print(image[tf.newaxis, ...].dtype)

print(image)

# dtype uint8 ->float형으로 바꿔줘야 모델 pred를 뽑을 수 있음
image = image.astype('float32')

# 이렇게하지 않으면 input pipeline의 uint8이 output model pred의 float와 호환되지 않아 오류가발생했었음
pred = model.predict(image[tf.newaxis, ...])

pred
pred = np.argmax(pred)

pred
preds = []



for fname in tqdm_notebook(sample_df['id']):

    path = os.path.join(test_dir, fname)



    img_pil = Image.open(path)

    image = np.array(img_pil)

    image = image.astype('float32') # 새로 추가



    pred = model.predict(image[tf.newaxis, ...])

    pred = np.argmax(pred)

    preds.append(pred)
submission_df = pd.DataFrame(data={'id': sample_df['id'], 'has_cactus': preds})
submission_df.head()
submission_df.to_csv('submission.csv', index=False)
os.listdir()