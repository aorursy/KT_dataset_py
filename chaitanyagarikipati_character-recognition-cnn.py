# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import random

import datetime

import matplotlib.pyplot as plt

from IPython.display import Image

import functools

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
tf.enable_eager_execution()
TRAIN_CSV_PATH = '/kaggle/input/street-view-getting-started-with-julia/trainLabels.csv'

TRAIN_IMGS_BASE_PATH = '/kaggle/input/street-view-getting-started-with-julia/trainresized/trainResized/'

TEST_IMGS_BASE_PATH = '/kaggle/input/street-view-getting-started-with-julia/testresized/testResized/'

BATH_SIZE = 256
train_data = pd.read_csv(TRAIN_CSV_PATH)
train_data.head()
LABELS = train_data['Class']

UNIQUE_LABELS = list(set(LABELS))
LABEL_IDX = [UNIQUE_LABELS.index(l) for l in LABELS]

LABEL_IDX = np.array(LABEL_IDX, dtype=np.float32)

train_data['label'] = LABEL_IDX
train_data.drop([283, 2289, 3135], inplace=True)

train_data.reset_index(inplace=True)
random_id = random.choice(train_data['ID'].values)

sample_img = TRAIN_IMGS_BASE_PATH + str(random_id) + '.Bmp'
!find {sample_img}
img_cnt = tf.read_file(filename=sample_img)

img = tf.io.decode_bmp(img_cnt, channels=3)

print(img.shape)

plt.imshow(img)

plt.title(LABELS[random_id-1])

print(LABEL_IDX[random_id - 1])

print(UNIQUE_LABELS.index(LABELS[random_id-1]))
train_data['img'] = [TRAIN_IMGS_BASE_PATH + str(id) + '.Bmp' for id in train_data['ID'].values]
train_data.head()
def transform_img(img, label=None):

    img_cnt = tf.read_file(img)

    img_cnt = tf.io.decode_bmp(img_cnt, channels=3)

#     img_cnt = tf.keras.applications.resnet50.preprocess_input(img_cnt)

    img_cnt /= 255

#     mean = tf.math.reduce_mean(img_cnt)

#     std = tf.math.reduce_std(img_cnt)

#     img_cnt = (img_cnt - std) / mean

    return img_cnt, label
def get_dataset(imgs, labels=None):

    dataset = (

        tf.data.Dataset.from_tensor_slices((imgs, labels))

        .shuffle(len(imgs))

        .map(transform_img)

        .batch(BATH_SIZE)

        .repeat()

        .prefetch(1)

    )

    iterator = dataset.make_one_shot_iterator()

    return iterator
X_train, X_test, y_train, y_test = train_test_split(train_data['img'], train_data['label'], test_size=0.2, random_state=42)
steps_per_epoch, validation_steps = X_train.shape[0]/BATH_SIZE, X_test.shape[0]/BATH_SIZE
validation_steps
train_iter = get_dataset(X_train, y_train)
validation_iter = get_dataset(X_test, y_test)
fig = plt.figure()

for i in range(1, 5):

    plt.subplot(5, 5, i)

    imgs, lbs = train_iter.get_next()

#     print(imgs.numpy().shape)

#     print(lbs.numpy().shape)

    plt.imshow(imgs[3])

    plt.title(UNIQUE_LABELS[int(lbs[3])])

    

plt.show()
Activation = 'elu'

Input = tf.keras.layers.Input

Conv2D = functools.partial(

        tf.keras.layers.Conv2D,

        activation=Activation,

        padding='same'

        )

Dense = functools.partial(

        tf.keras.layers.Dense

        )

Dropout = tf.keras.layers.Dropout

Avgpool = tf.keras.layers.AveragePooling2D

MaxPool2D = tf.keras.layers.MaxPool2D

BatchNorm = tf.keras.layers.BatchNormalization

Flatten = tf.keras.layers.Flatten
def get_model(outputs_shape):

    input = Input(shape=(20, 20, 3,))

    conv_1 = Conv2D(16, (2, 2))(input)

    conv_2 = Conv2D(16, (2, 2))(conv_1)

    conv_3 = Conv2D(32, (3, 3))(conv_2)

    avg_1 = Avgpool((2, 2))(conv_2)

    batch_norm_2 = BatchNorm()(conv_2)

    

    conv_3 = Conv2D(64, (3, 3))(batch_norm_2)

    conv_4 = Conv2D(64, (3, 3))(conv_3)

#     avg_2 = Avgpool((2, 2))(conv_4)

    batch_norm_4 =  BatchNorm()(conv_4)

    

    conv_5 = Conv2D(32, (3, 3))(batch_norm_4)

    conv_6 = Conv2D(32, (5, 5))(conv_5)

    dropout_1 = Dropout(0.3)(conv_6)

    batch_norm_6 =  BatchNorm()(dropout_1)

    

    conv_7 = Conv2D(16, (5, 5))(batch_norm_6)

    conv_8 = Conv2D(16, (5, 5))(conv_7)

    batch_norm_7 =  BatchNorm()(conv_8)

    

    flat_1 = Flatten()(batch_norm_7)

    dense_1 = Dense(512, activation=Activation)(flat_1)

    outputs = Dense(outputs_shape, activation='softmax')(dense_1)

    

    model = tf.keras.Model(input, outputs)

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy' ,metrics=['accuracy'])

    model.summary()

    return model
unique_labels_count = len(list(set(LABELS)))

print(unique_labels_count)

model = get_model(unique_labels_count)
logdir = os.path.join("/tmp/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True),

    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),

    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

]
history = model.fit(train_iter, steps_per_epoch=20, epochs=50, validation_data=validation_iter, validation_steps=5, callbacks=callbacks)
test_imgs = []

for dirname, _, filenames in os.walk(TEST_IMGS_BASE_PATH):

    for filename in filenames:

        test_imgs.append(os.path.join(dirname, filename))

print(test_imgs[:5])

test_imgs = np.array(test_imgs)
len(test_imgs)/256
grey_imgs = []

for i, img in enumerate(test_imgs):

    try:

        img_cnt = tf.read_file(img)

        img_cnt = tf.image.decode_bmp(img_cnt, channels=3)

    except:

        print(i)

        grey_imgs.append(i)
test_imgs = np.delete(test_imgs, grey_imgs)
test_pipeline = get_dataset(test_imgs, tf.zeros(len(test_imgs)))
predictions = model.predict(test_pipeline, steps=25)
rand_img = random.choice(range(len(test_imgs)))

img_cnt = tf.read_file(test_imgs[rand_img])

img_cnt = tf.image.decode_bmp(img_cnt)

plt.imshow(img_cnt)

plt.title(UNIQUE_LABELS[np.argmax(predictions[rand_img])])